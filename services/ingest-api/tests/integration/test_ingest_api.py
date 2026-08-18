from __future__ import annotations

import asyncio
import hashlib
import io
import os
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import boto3
import pytest
import redis
from celery import Celery
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from voice_pipeline_task_client import TaskPublisher
from voice_pipeline_task_contracts import DIARIZE_AUDIO_PART

from voice_pipeline_ingest_api.app import create_app
from voice_pipeline_ingest_api.config import (
    ApplicationSettings,
    EnvironmentSettings,
    IngestPolicy,
    PolicySettings,
)
from voice_pipeline_ingest_api.dependencies import (
    ServiceResources,
    normalize_database_url,
)
from voice_pipeline_ingest_api.repository import (
    DuplicateContentError,
    RawAudioCreate,
    RawAudioRepository,
    RawAudioRepositoryError,
)
from voice_pipeline_ingest_api.services.storage import ObjectStorage

pytestmark = pytest.mark.integration
TEST_CELERY_QUEUE = "voice-pipeline-ingest-integration"

REQUIRED_TEST_ENVIRONMENT = (
    "TEST_DATABASE_URL",
    "TEST_CELERY_BROKER_URL",
    "TEST_S3_BUCKET",
    "TEST_S3_REGION",
    "TEST_S3_ENDPOINT_URL",
    "TEST_AWS_ACCESS_KEY_ID",
    "TEST_AWS_SECRET_ACCESS_KEY",
)


def _require_test_environment() -> dict[str, str]:
    missing = [name for name in REQUIRED_TEST_ENVIRONMENT if not os.getenv(name)]
    if missing:
        pytest.skip("External integration services are not configured.")
    return {name: os.environ[name] for name in REQUIRED_TEST_ENVIRONMENT}


def _source_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44_100)
        wav_file.writeframes(b"\x00" * 4_410 * 2 * 2)
    return output.getvalue()


def _settings(environment: dict[str, str]) -> ApplicationSettings:
    return ApplicationSettings(
        policy=PolicySettings(
            ingest=IngestPolicy(
                max_upload_bytes=1_048_576,
                max_concurrent_requests=10,
            )
        ),
        environment=EnvironmentSettings(
            database_url=environment["TEST_DATABASE_URL"],
            celery_broker_url=environment["TEST_CELERY_BROKER_URL"],
            s3_bucket=environment["TEST_S3_BUCKET"],
            s3_region=environment["TEST_S3_REGION"],
            s3_endpoint_url=environment["TEST_S3_ENDPOINT_URL"],
        ),
    )


def _s3_client(environment: dict[str, str]) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=environment["TEST_S3_ENDPOINT_URL"],
        region_name=environment["TEST_S3_REGION"],
        aws_access_key_id=environment["TEST_AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=environment["TEST_AWS_SECRET_ACCESS_KEY"],
    )


def _test_task_publisher(environment: dict[str, str]) -> TaskPublisher:
    celery_app = Celery(
        "voice-pipeline-ingest-integration",
        broker=environment["TEST_CELERY_BROKER_URL"],
    )
    celery_app.conf.update(
        accept_content=["json"],
        task_serializer="json",
        result_serializer="json",
        task_default_queue=TEST_CELERY_QUEUE,
    )
    return TaskPublisher(celery_app)


def _test_resource_factory(
    environment: dict[str, str],
) -> Callable[[ApplicationSettings], ServiceResources]:
    def create_resources(settings: ApplicationSettings) -> ServiceResources:
        engine = create_engine(
            normalize_database_url(settings.environment.database_url),
            pool_pre_ping=True,
        )
        return ServiceResources(
            engine=engine,
            session_factory=sessionmaker(
                bind=engine,
                expire_on_commit=False,
            ),
            storage=ObjectStorage(
                client=_s3_client(environment),
                bucket=settings.environment.s3_bucket,
            ),
            task_publisher=_test_task_publisher(environment),
        )

    return create_resources


@pytest.fixture
def integration_environment() -> Iterator[dict[str, str]]:
    environment = _require_test_environment()
    database_url = environment["TEST_DATABASE_URL"]
    engine = create_engine(normalize_database_url(database_url))
    schema_path = Path(__file__).parents[4] / "schema" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    if not inspect(engine).has_table("raw_audios"):
        with engine.begin() as connection:
            connection.exec_driver_sql(schema_sql)
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE raw_audios CASCADE"))

    redis_client = redis.Redis.from_url(environment["TEST_CELERY_BROKER_URL"])
    redis_client.flushdb()

    s3_client = _s3_client(environment)
    bucket = environment["TEST_S3_BUCKET"]
    s3_client.head_bucket(Bucket=bucket)

    yield environment

    listed = s3_client.list_objects_v2(Bucket=bucket)
    for item in listed.get("Contents", []):
        s3_client.delete_object(Bucket=bucket, Key=item["Key"])
    redis_client.flushdb()
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE raw_audios CASCADE"))
    engine.dispose()


def test_complete_ingest_deduplication_and_query(integration_environment) -> None:
    async def scenario() -> None:
        environment = integration_environment
        app = create_app(
            _settings(environment),
            resource_factory=_test_resource_factory(environment),
        )
        source_wav = _source_wav()
        transport = ASGITransport(app=app)

        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=transport, base_url="http://test") as client,
        ):
            first = await client.post(
                "/v1/raw-audios",
                files={"audio": ("episode.wav", source_wav, "audio/wav")},
                data={
                    "title": "Integration episode",
                    "source_url": "https://example.test/episode",
                    "lang": "en",
                    "meta": '{"feed":"integration"}',
                },
            )
            duplicate = await client.post(
                "/v1/raw-audios",
                files={"audio": ("renamed.wav", source_wav, "audio/wav")},
            )
            raw_audio_id = first.json()["id"]
            fetched = await client.get(f"/v1/raw-audios/{raw_audio_id}")

        assert first.status_code == 202
        assert first.json()["task_id"] is not None
        assert duplicate.status_code == 200
        assert duplicate.json() == {
            "id": raw_audio_id,
            "status": "pending",
            "content_sha1": hashlib.sha1(
                source_wav,
                usedforsecurity=False,
            ).hexdigest(),
            "task_id": None,
            "deduplicated": True,
        }
        assert fetched.status_code == 200
        assert fetched.json()["title"] == "Integration episode"
        assert fetched.json()["meta"] == {"feed": "integration"}
        assert fetched.json()["duration_ms"] == 100

        engine = create_engine(normalize_database_url(environment["TEST_DATABASE_URL"]))
        with engine.connect() as connection:
            count = connection.scalar(text("SELECT count(*) FROM raw_audios"))
            stored = connection.execute(
                text(
                    "SELECT status, content_sha1, audio_uri, duration_ms, size_bytes "
                    "FROM raw_audios"
                )
            ).one()
        engine.dispose()
        assert count == 1
        assert stored.status == "pending"
        assert (
            stored.content_sha1
            == hashlib.sha1(
                source_wav,
                usedforsecurity=False,
            ).hexdigest()
        )
        assert stored.audio_uri.endswith(f"raw_audios/{raw_audio_id}/audio.wav")
        assert stored.duration_ms == 100
        assert stored.size_bytes > 44

        s3_client = _s3_client(environment)
        objects = s3_client.list_objects_v2(Bucket=environment["TEST_S3_BUCKET"])
        keys = [item["Key"] for item in objects.get("Contents", [])]
        assert keys == [f"raw_audios/{raw_audio_id}/audio.wav"]
        normalized = s3_client.get_object(
            Bucket=environment["TEST_S3_BUCKET"],
            Key=keys[0],
        )["Body"].read()
        with wave.open(io.BytesIO(normalized), "rb") as wav_file:
            assert wav_file.getframerate() == 16_000
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getcomptype() == "NONE"

    asyncio.run(scenario())


def test_concurrent_identical_uploads_leave_one_complete_ingest(
    integration_environment,
) -> None:
    async def scenario() -> None:
        environment = integration_environment
        app = create_app(
            _settings(environment),
            resource_factory=_test_resource_factory(environment),
        )
        source_wav = _source_wav()
        transport = ASGITransport(app=app)

        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=transport, base_url="http://test") as client,
        ):
            first, second = await asyncio.gather(
                client.post(
                    "/v1/raw-audios",
                    files={"audio": ("first.wav", source_wav, "audio/wav")},
                ),
                client.post(
                    "/v1/raw-audios",
                    files={"audio": ("second.wav", source_wav, "audio/wav")},
                ),
            )

        assert sorted([first.status_code, second.status_code]) == [200, 202]
        assert first.json()["id"] == second.json()["id"]
        assert sorted(
            [first.json()["deduplicated"], second.json()["deduplicated"]]
        ) == [False, True]
        task_ids = [first.json()["task_id"], second.json()["task_id"]]
        assert sum(task_id is not None for task_id in task_ids) == 1

        engine = create_engine(normalize_database_url(environment["TEST_DATABASE_URL"]))
        with engine.connect() as connection:
            count = connection.scalar(text("SELECT count(*) FROM raw_audios"))
        engine.dispose()
        assert count == 1

        s3_client = _s3_client(environment)
        objects = s3_client.list_objects_v2(Bucket=environment["TEST_S3_BUCKET"])
        keys = [item["Key"] for item in objects.get("Contents", [])]
        assert keys == [f"raw_audios/{first.json()['id']}/audio.wav"]

    asyncio.run(scenario())


def test_repository_unique_conflict_and_failed_status(integration_environment) -> None:
    engine = create_engine(
        normalize_database_url(integration_environment["TEST_DATABASE_URL"])
    )
    repository = RawAudioRepository(
        sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    )
    first_id = UUID("11111111-1111-1111-1111-111111111111")
    second_id = UUID("22222222-2222-2222-2222-222222222222")

    def values(raw_audio_id: UUID) -> RawAudioCreate:
        return RawAudioCreate(
            id=raw_audio_id,
            audio_uri=f"s3://test-bucket/raw_audios/{raw_audio_id}/audio.wav",
            content_sha1="b" * 40,
            title=None,
            source_url=None,
            lang="en",
            meta={},
            duration_ms=100,
            size_bytes=3_244,
        )

    repository.create(values(first_id))
    with pytest.raises(DuplicateContentError) as duplicate:
        repository.create(values(second_id))

    assert duplicate.value.record.id == first_id
    repository.mark_failed(first_id, "Failed to publish the split_raw_audio_into_parts task.")
    failed = repository.get(first_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error == "Failed to publish the split_raw_audio_into_parts task."

    for advanced_status in ("splitting", "split_completed"):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE raw_audios "
                    "SET status = :status, error = NULL "
                    "WHERE id = :raw_audio_id"
                ),
                {"status": advanced_status, "raw_audio_id": first_id},
            )
        with pytest.raises(
            RawAudioRepositoryError, match="missing or no longer pending"
        ):
            repository.mark_failed(first_id, "publication outcome unknown")
        advanced = repository.get(first_id)
        assert advanced is not None
        assert advanced.status == advanced_status
        assert advanced.error is None
    engine.dispose()


def test_trigger_registered_task_publishes_uuid_to_real_broker(
    integration_environment,
) -> None:
    async def scenario() -> None:
        environment = integration_environment
        app = create_app(
            _settings(environment),
            resource_factory=_test_resource_factory(environment),
        )
        entity_id = UUID("33333333-3333-3333-3333-333333333333")
        transport = ASGITransport(app=app)

        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=transport, base_url="http://test") as client,
        ):
            response = await client.post(
                "/v1/tasks/trigger",
                json={
                    "task_name": DIARIZE_AUDIO_PART.name,
                    "id": str(entity_id),
                },
            )

        assert response.status_code == 202
        assert response.json()["task_name"] == DIARIZE_AUDIO_PART.name
        assert response.json()["id"] == str(entity_id)
        assert response.json()["task_id"]

        broker_app = Celery(
            "voice-pipeline-ingest-integration-reader",
            broker=environment["TEST_CELERY_BROKER_URL"],
        )
        try:
            with (
                broker_app.connection_for_read() as connection,
                connection.SimpleQueue(DIARIZE_AUDIO_PART.queue) as queue,
            ):
                message = queue.get(block=True, timeout=5)
                assert message.headers["task"] == DIARIZE_AUDIO_PART.name
                assert message.payload[0] == [str(entity_id)]
                assert message.headers["id"] == response.json()["task_id"]
                message.ack()
        finally:
            broker_app.close()

    asyncio.run(scenario())
