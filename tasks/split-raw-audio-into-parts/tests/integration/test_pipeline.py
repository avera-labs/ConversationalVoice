from __future__ import annotations

import io
import json
import os
import struct
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import boto3
import psycopg
import pytest
import redis
from botocore.config import Config
from celery import Celery
from dotenv import load_dotenv
from psycopg.rows import dict_row
from voice_pipeline_task_contracts import DIARIZE_AUDIO_PART

from voice_pipeline_split_raw_audio_into_parts.config import (
    EnvironmentSettings,
    WindowingPolicy,
)
from voice_pipeline_split_raw_audio_into_parts.errors import FailureReason
from voice_pipeline_split_raw_audio_into_parts.publisher import (
    DiarizationPublisher,
)
from voice_pipeline_split_raw_audio_into_parts.repository import SplitRepository
from voice_pipeline_split_raw_audio_into_parts.storage import ObjectStorage
from voice_pipeline_split_raw_audio_into_parts.task import (
    SplitRawAudioIntoPartsHandler,
    TaskStageError,
)
from voice_pipeline_split_raw_audio_into_parts.vad import VadResult
from voice_pipeline_split_raw_audio_into_parts.wav_io import (
    CHANNEL_COUNT,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
)
from voice_pipeline_split_raw_audio_into_parts.windowing import FrameSpan


pytestmark = pytest.mark.integration

if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Set RUN_INTEGRATION_TESTS=1 through tests/integration/run.sh.",
        allow_module_level=True,
    )


@dataclass(frozen=True, slots=True, repr=False)
class IntegrationConfiguration:
    settings: EnvironmentSettings

    def __repr__(self) -> str:
        return "<IntegrationConfiguration redacted>"


@dataclass(slots=True, repr=False)
class IntegrationServices:
    settings: EnvironmentSettings
    s3: Any
    broker_app: Celery
    redis: redis.Redis
    raw_audio_ids: set[UUID] = field(default_factory=set)

    def __repr__(self) -> str:
        return "<IntegrationServices redacted>"

    def fetch_all(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        with psycopg.connect(
            _database_connection_url(self.settings),
            row_factory=dict_row,
        ) as connection:
            return list(connection.execute(query, parameters).fetchall())

    def clear_owned_task_messages(self) -> None:
        if not self.raw_audio_ids:
            return
        rows = self.fetch_all(
            """
            SELECT id
            FROM audio_parts
            WHERE raw_audio_id = ANY(%s)
            """,
            (list(self.raw_audio_ids),),
        )
        owned_ids = {str(row["id"]) for row in rows}
        message_count = self.task_queue_size()
        if message_count == 0:
            return
        retained = []
        with self.broker_app.connection_for_read() as connection:
            with connection.SimpleQueue(DIARIZE_AUDIO_PART.queue) as queue:
                for _ in range(message_count):
                    message = queue.get(block=True, timeout=5)
                    identifier = str(message.payload[0][0])
                    if identifier in owned_ids:
                        message.ack()
                    else:
                        retained.append(message)
                for message in retained:
                    message.reject(requeue=True)

    def task_queue_size(self) -> int:
        return int(self.redis.llen(DIARIZE_AUDIO_PART.queue))


@dataclass(slots=True)
class RealAdapters:
    repository: SplitRepository
    storage: ObjectStorage
    publisher: DiarizationPublisher
    policy: WindowingPolicy

    def handler(
        self,
        vad: Any,
        *,
        repository: Any | None = None,
        storage: Any | None = None,
        publisher: Any | None = None,
    ) -> SplitRawAudioIntoPartsHandler:
        return SplitRawAudioIntoPartsHandler(
            repository=repository or self.repository,
            storage=storage or self.storage,
            vad=vad,
            publisher=publisher or self.publisher,
            windowing_policy=self.policy,
        )

    def close(self) -> None:
        self.publisher.close()
        self.storage.close()
        self.repository.close()


class DeterministicVad:
    def __init__(self, segments: tuple[FrameSpan, ...]) -> None:
        self.segments = segments
        self.calls = 0
        self._lock = threading.Lock()

    def run(self, audio_path: Path) -> VadResult:
        assert audio_path.is_file()
        with self._lock:
            self.calls += 1
        return VadResult(
            model="pyannote/segmentation-3.0",
            audio_frame_count=SAMPLE_RATE,
            segments=self.segments,
        )


class BlockingVad(DeterministicVad):
    def __init__(self, segments: tuple[FrameSpan, ...]) -> None:
        super().__init__(segments)
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, audio_path: Path) -> VadResult:
        self.started.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("integration VAD release timed out")
        return super().run(audio_path)


class FailSecondPartStorage:
    def __init__(self, delegate: ObjectStorage) -> None:
        self.delegate = delegate
        self.part_calls = 0

    def download_raw_audio(self, audio_uri: str, destination: Path) -> None:
        self.delegate.download_raw_audio(audio_uri, destination)

    def upload_vad_segments(self, raw_audio_id: UUID, path: Path) -> str:
        return self.delegate.upload_vad_segments(raw_audio_id, path)

    def upload_audio_part(
        self,
        raw_audio_id: UUID,
        part_index: int,
        path: Path,
    ) -> str:
        self.part_calls += 1
        if self.part_calls == 2:
            raise RuntimeError("injected second-part upload failure")
        return self.delegate.upload_audio_part(raw_audio_id, part_index, path)


class FailPersistenceRepository:
    def __init__(self, delegate: SplitRepository) -> None:
        self.delegate = delegate

    def claim(self, raw_audio_id: UUID) -> Any:
        return self.delegate.claim(raw_audio_id)

    def persist_parts_and_complete(
        self,
        raw_audio_id: UUID,
        drafts: list[Any],
    ) -> list[Any]:
        raise RuntimeError("injected persistence failure")

    def list_pending_audio_part_ids(self, raw_audio_id: UUID) -> list[UUID]:
        return self.delegate.list_pending_audio_part_ids(raw_audio_id)

    def count_audio_parts(self, raw_audio_id: UUID) -> int:
        return self.delegate.count_audio_parts(raw_audio_id)

    def mark_failed(self, raw_audio_id: UUID, error: str) -> None:
        self.delegate.mark_failed(raw_audio_id, error)


class FailSecondPublication:
    def __init__(self, delegate: DiarizationPublisher) -> None:
        self.delegate = delegate
        self.calls = 0

    def publish(self, audio_part_id: UUID) -> str:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("injected second-publication failure")
        return self.delegate.publish(audio_part_id)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing integration environment variable: {name}.")
    return value


def _database_connection_url(settings: EnvironmentSettings) -> str:
    return settings.database_url.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )


@pytest.fixture(scope="session")
def integration_configuration() -> IntegrationConfiguration:
    env_file = Path(__file__).parents[4] / ".env.test"
    load_dotenv(dotenv_path=env_file, override=True)
    os.environ["AWS_ACCESS_KEY_ID"] = _required_environment(
        "TEST_AWS_ACCESS_KEY_ID"
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = _required_environment(
        "TEST_AWS_SECRET_ACCESS_KEY"
    )
    settings = EnvironmentSettings(
        database_url=_required_environment("TEST_DATABASE_URL"),
        celery_broker_url=_required_environment("TEST_CELERY_BROKER_URL"),
        s3_bucket=_required_environment("TEST_S3_BUCKET"),
        s3_region=_required_environment("TEST_S3_REGION"),
        s3_endpoint_url=_required_environment("TEST_S3_ENDPOINT_URL"),
        hf_token=_required_environment("TEST_HF_TOKEN"),
    )
    return IntegrationConfiguration(settings=settings)


@pytest.fixture(scope="session")
def services(
    integration_configuration: IntegrationConfiguration,
) -> IntegrationServices:
    environment_settings = integration_configuration.settings
    s3 = boto3.client(
        "s3",
        region_name=environment_settings.s3_region,
        endpoint_url=environment_settings.s3_endpoint_url,
        config=Config(s3={"addressing_style": "path"}),
    )
    broker_app = Celery(
        "split-integration-services",
        broker=environment_settings.celery_broker_url,
    )
    redis_client = redis.Redis.from_url(
        environment_settings.celery_broker_url,
        decode_responses=False,
    )

    try:
        with psycopg.connect(_database_connection_url(environment_settings)) as connection:
            tables = connection.execute(
                """
                SELECT to_regclass('raw_audios'),
                       to_regclass('audio_parts'),
                       to_regclass('chunks')
                """
            ).fetchone()
        if tables is None or any(value is None for value in tables):
            raise RuntimeError(
                "The integration database does not contain the OSS schema."
            )

        with broker_app.connection_for_read() as connection:
            connection.ensure_connection(max_retries=0)
        s3.head_bucket(Bucket=environment_settings.s3_bucket)
    except Exception:
        redis_client.close()
        broker_app.close()
        s3.close()
        raise RuntimeError(
            "The .env.test integration services are not ready."
        ) from None

    value = IntegrationServices(
        settings=environment_settings,
        s3=s3,
        broker_app=broker_app,
        redis=redis_client,
    )
    if value.task_queue_size() != 0:
        redis_client.close()
        broker_app.close()
        s3.close()
        raise RuntimeError(
            "The integration diarization queue must be empty before testing."
        )
    yield value
    redis_client.close()
    broker_app.close()
    s3.close()


@pytest.fixture(autouse=True)
def reset_integration_state(services: IntegrationServices):
    if services.task_queue_size() != 0:
        raise RuntimeError(
            "The integration diarization queue is not isolated."
        )
    yield
    services.clear_owned_task_messages()

    raw_audio_ids = list(services.raw_audio_ids)
    if raw_audio_ids:
        with psycopg.connect(_database_connection_url(services.settings)) as connection:
            connection.execute(
                "DELETE FROM raw_audios WHERE id = ANY(%s)",
                (raw_audio_ids,),
            )

    for raw_audio_id in raw_audio_ids:
        response = services.s3.list_objects_v2(
            Bucket=services.settings.s3_bucket,
            Prefix=f"raw_audios/{raw_audio_id}/",
        )
        objects = [
            {"Key": item["Key"]}
            for item in response.get("Contents", [])
        ]
        if objects:
            services.s3.delete_objects(
                Bucket=services.settings.s3_bucket,
                Delete={"Objects": objects},
            )
    services.raw_audio_ids.clear()


@pytest.fixture
def adapters(
    services: IntegrationServices,
) -> RealAdapters:
    value = RealAdapters(
        repository=SplitRepository.create(services.settings),
        storage=ObjectStorage.create(services.settings),
        publisher=DiarizationPublisher.create(services.settings),
        policy=WindowingPolicy(
            gap_threshold_ms=0,
            min_window_ms=100,
            max_window_ms=1_000,
            pad_before_ms=0,
            pad_after_ms=0,
        ),
    )
    yield value
    value.close()


def _wav_bytes() -> bytes:
    payload = struct.pack(
        f"<{SAMPLE_RATE}h",
        *[index % 1_000 for index in range(SAMPLE_RATE)],
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(CHANNEL_COUNT)
        writer.setsampwidth(SAMPLE_WIDTH_BYTES)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(payload)
    return output.getvalue()


def _seed_raw_audio(
    services: IntegrationServices,
    *,
    status: str = "pending",
) -> UUID:
    raw_audio_id = uuid4()
    services.raw_audio_ids.add(raw_audio_id)
    key = f"raw_audios/{raw_audio_id}/audio.wav"
    services.s3.put_object(
        Bucket=services.settings.s3_bucket,
        Key=key,
        Body=_wav_bytes(),
        ContentType="audio/wav",
    )
    audio_uri = f"s3://{services.settings.s3_bucket}/{key}"
    with psycopg.connect(_database_connection_url(services.settings)) as connection:
        connection.execute(
            """
            INSERT INTO raw_audios (id, status, audio_uri, lang)
            VALUES (%s, %s, %s, 'en')
            """,
            (raw_audio_id, status, audio_uri),
        )
    return raw_audio_id


def _read_tasks(
    broker_url: str,
    expected_count: int,
) -> list[tuple[str, str]]:
    app = Celery("split-integration-reader", broker=broker_url)
    received: list[tuple[str, str]] = []
    try:
        with app.connection_for_read() as connection:
            with connection.SimpleQueue(DIARIZE_AUDIO_PART.queue) as queue:
                for _ in range(expected_count):
                    message = queue.get(block=True, timeout=5)
                    received.append(
                        (
                            message.headers["task"],
                            str(message.payload[0][0]),
                        )
                    )
                    message.ack()
    finally:
        app.close()
    return received


def _two_segments() -> tuple[FrameSpan, ...]:
    return (
        FrameSpan(1_600, 4_800),
        FrameSpan(8_000, 11_200),
    )


def test_real_services_end_to_end_and_completed_redelivery(
    services: IntegrationServices,
    adapters: RealAdapters,
) -> None:
    raw_audio_id = _seed_raw_audio(services)
    vad = DeterministicVad(_two_segments())
    handler = adapters.handler(vad)

    first = handler(str(raw_audio_id))

    assert first["status"] == "split_completed"
    assert first["audio_part_count"] == 2
    assert first["diarization_dispatch_count"] == 2
    assert vad.calls == 1

    raw_rows = services.fetch_all(
        "SELECT status, error FROM raw_audios WHERE id = %s",
        (raw_audio_id,),
    )
    assert raw_rows == [{"status": "split_completed", "error": None}]

    part_rows = services.fetch_all(
        """
        SELECT id, part_index, status, audio_uri, lang,
               relative_start_ms, relative_end_ms, duration_ms
        FROM audio_parts
        WHERE raw_audio_id = %s
        ORDER BY part_index
        """,
        (raw_audio_id,),
    )
    assert [
        (
            row["part_index"],
            row["status"],
            row["lang"],
            row["relative_start_ms"],
            row["relative_end_ms"],
            row["duration_ms"],
        )
        for row in part_rows
    ] == [
        (0, "pending", "en", 100, 300, 200),
        (1, "pending", "en", 500, 700, 200),
    ]

    keys = {
        item["Key"]
        for item in services.s3.list_objects_v2(
            Bucket=services.settings.s3_bucket
        ).get("Contents", [])
    }
    assert keys == {
        f"raw_audios/{raw_audio_id}/audio.wav",
        f"raw_audios/{raw_audio_id}/vad_segments.json",
        f"raw_audios/{raw_audio_id}/audio_parts/0/audio.wav",
        f"raw_audios/{raw_audio_id}/audio_parts/1/audio.wav",
    }

    artifact = services.s3.get_object(
        Bucket=services.settings.s3_bucket,
        Key=f"raw_audios/{raw_audio_id}/vad_segments.json",
    )
    document = json.loads(artifact["Body"].read())
    assert [item["start_ms"] for item in document["segments"]] == [100, 500]

    for index in (0, 1):
        response = services.s3.get_object(
            Bucket=services.settings.s3_bucket,
            Key=f"raw_audios/{raw_audio_id}/audio_parts/{index}/audio.wav",
        )
        with wave.open(io.BytesIO(response["Body"].read()), "rb") as reader:
            assert reader.getframerate() == SAMPLE_RATE
            assert reader.getnchannels() == CHANNEL_COUNT
            assert reader.getsampwidth() == SAMPLE_WIDTH_BYTES
            assert reader.getnframes() == 3_200

    first_messages = _read_tasks(services.settings.celery_broker_url, 2)
    assert {task_name for task_name, _ in first_messages} == {
        DIARIZE_AUDIO_PART.name
    }
    assert {UUID(identifier) for _, identifier in first_messages} == {
        row["id"] for row in part_rows
    }

    second = handler(str(raw_audio_id))

    assert second["status"] == "split_completed"
    assert second["audio_part_count"] == 2
    assert second["diarization_dispatch_count"] == 2
    assert vad.calls == 1
    assert len(
        services.fetch_all(
            "SELECT id FROM audio_parts WHERE raw_audio_id = %s",
            (raw_audio_id,),
        )
    ) == 2
    _read_tasks(services.settings.celery_broker_url, 2)


def test_concurrent_delivery_only_one_handler_runs_vad(
    services: IntegrationServices,
    adapters: RealAdapters,
) -> None:
    raw_audio_id = _seed_raw_audio(services)
    vad = BlockingVad((FrameSpan(1_600, 4_800),))
    handler = adapters.handler(vad)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(handler, str(raw_audio_id))
        assert vad.started.wait(timeout=5)
        second = handler(str(raw_audio_id))
        vad.release.set()
        first = first_future.result(timeout=10)

    assert first["status"] == "split_completed"
    assert second["status"] == "already_processing"
    assert vad.calls == 1
    assert len(
        services.fetch_all(
            "SELECT id FROM audio_parts WHERE raw_audio_id = %s",
            (raw_audio_id,),
        )
    ) == 1
    _read_tasks(services.settings.celery_broker_url, 1)


def test_zero_windows_persists_empty_vad_artifact(
    services: IntegrationServices,
    adapters: RealAdapters,
) -> None:
    raw_audio_id = _seed_raw_audio(services)
    handler = adapters.handler(DeterministicVad(()))

    result = handler(str(raw_audio_id))

    assert result["audio_part_count"] == 0
    assert result["diarization_dispatch_count"] == 0
    assert services.fetch_all(
        "SELECT status FROM raw_audios WHERE id = %s",
        (raw_audio_id,),
    ) == [{"status": "split_completed"}]
    assert services.fetch_all(
        "SELECT id FROM audio_parts WHERE raw_audio_id = %s",
        (raw_audio_id,),
    ) == []
    response = services.s3.get_object(
        Bucket=services.settings.s3_bucket,
        Key=f"raw_audios/{raw_audio_id}/vad_segments.json",
    )
    assert json.loads(response["Body"].read())["segments"] == []
    assert services.task_queue_size() == 0


def test_existing_part_is_reused_without_overwrite(
    services: IntegrationServices,
    adapters: RealAdapters,
) -> None:
    raw_audio_id = _seed_raw_audio(services)
    existing_id = uuid4()
    with psycopg.connect(_database_connection_url(services.settings)) as connection:
        connection.execute(
            """
            INSERT INTO audio_parts (
                id, raw_audio_id, part_index, status, audio_uri, lang,
                relative_start_ms, relative_end_ms, duration_ms
            )
            VALUES (%s, %s, 0, 'pending', %s, 'en', 10, 20, 10)
            """,
            (
                existing_id,
                raw_audio_id,
                (
                    f"s3://{services.settings.s3_bucket}/"
                    "existing/object.wav"
                ),
            ),
        )

    result = adapters.handler(
        DeterministicVad((FrameSpan(1_600, 4_800),))
    )(str(raw_audio_id))

    assert result["audio_part_count"] == 1
    rows = services.fetch_all(
        """
        SELECT id, audio_uri, relative_start_ms, relative_end_ms
        FROM audio_parts
        WHERE raw_audio_id = %s
        """,
        (raw_audio_id,),
    )
    assert rows == [
        {
            "id": existing_id,
            "audio_uri": (
                f"s3://{services.settings.s3_bucket}/existing/object.wav"
            ),
            "relative_start_ms": 10,
            "relative_end_ms": 20,
        }
    ]
    messages = _read_tasks(services.settings.celery_broker_url, 1)
    assert UUID(messages[0][1]) == existing_id


def test_partial_upload_failure_retries_to_complete(
    services: IntegrationServices,
    adapters: RealAdapters,
) -> None:
    raw_audio_id = _seed_raw_audio(services)
    vad = DeterministicVad(_two_segments())
    failing_storage = FailSecondPartStorage(adapters.storage)

    with pytest.raises(TaskStageError) as failure:
        adapters.handler(vad, storage=failing_storage)(str(raw_audio_id))

    assert failure.value.reason is FailureReason.UPLOAD_FAILED
    assert services.fetch_all(
        "SELECT status FROM raw_audios WHERE id = %s",
        (raw_audio_id,),
    ) == [{"status": "failed"}]
    assert services.fetch_all(
        "SELECT id FROM audio_parts WHERE raw_audio_id = %s",
        (raw_audio_id,),
    ) == []

    result = adapters.handler(vad)(str(raw_audio_id))

    assert result["status"] == "split_completed"
    assert result["audio_part_count"] == 2
    assert vad.calls == 2
    _read_tasks(services.settings.celery_broker_url, 2)


def test_persistence_failure_keeps_objects_and_retry_converges(
    services: IntegrationServices,
    adapters: RealAdapters,
) -> None:
    raw_audio_id = _seed_raw_audio(services)
    vad = DeterministicVad(_two_segments())
    failing_repository = FailPersistenceRepository(adapters.repository)

    with pytest.raises(TaskStageError) as failure:
        adapters.handler(vad, repository=failing_repository)(str(raw_audio_id))

    assert failure.value.reason is FailureReason.PERSISTENCE_FAILED
    assert services.fetch_all(
        "SELECT status FROM raw_audios WHERE id = %s",
        (raw_audio_id,),
    ) == [{"status": "failed"}]
    assert services.fetch_all(
        "SELECT id FROM audio_parts WHERE raw_audio_id = %s",
        (raw_audio_id,),
    ) == []
    keys = {
        item["Key"]
        for item in services.s3.list_objects_v2(
            Bucket=services.settings.s3_bucket
        ).get("Contents", [])
    }
    assert f"raw_audios/{raw_audio_id}/audio_parts/1/audio.wav" in keys

    result = adapters.handler(vad)(str(raw_audio_id))

    assert result["status"] == "split_completed"
    assert result["audio_part_count"] == 2
    _read_tasks(services.settings.celery_broker_url, 2)


def test_partial_dispatch_is_at_least_once_after_retry(
    services: IntegrationServices,
    adapters: RealAdapters,
) -> None:
    raw_audio_id = _seed_raw_audio(services)
    vad = DeterministicVad(_two_segments())
    failing_publisher = FailSecondPublication(adapters.publisher)

    with pytest.raises(TaskStageError) as failure:
        adapters.handler(vad, publisher=failing_publisher)(str(raw_audio_id))

    assert failure.value.reason is FailureReason.DOWNSTREAM_DISPATCH_FAILED
    assert services.fetch_all(
        "SELECT status FROM raw_audios WHERE id = %s",
        (raw_audio_id,),
    ) == [{"status": "failed"}]
    part_rows = services.fetch_all(
        "SELECT id FROM audio_parts WHERE raw_audio_id = %s ORDER BY part_index",
        (raw_audio_id,),
    )
    assert len(part_rows) == 2
    first_delivery = _read_tasks(services.settings.celery_broker_url, 1)

    result = adapters.handler(vad)(str(raw_audio_id))

    assert result["status"] == "split_completed"
    assert result["audio_part_count"] == 2
    retry_deliveries = _read_tasks(services.settings.celery_broker_url, 2)
    assert first_delivery[0][1] in {
        identifier for _, identifier in retry_deliveries
    }
