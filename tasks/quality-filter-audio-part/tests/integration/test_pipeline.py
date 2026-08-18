from __future__ import annotations

import io
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

import boto3
import numpy as np
import psycopg
import pytest
from celery import Celery
from dotenv import load_dotenv
from voice_pipeline_diarization_artifact import RawTurn, build_artifact
from voice_pipeline_task_contracts import QUALITY_FILTER_AUDIO_PART

from voice_pipeline_quality_filter_audio_part.config import (
    EnvironmentSettings,
    PlannerPolicy,
    QualityPolicy,
    TaskPolicy,
)
from voice_pipeline_quality_filter_audio_part.errors import ErrorCode, TaskStageError
from voice_pipeline_quality_filter_audio_part.intervals import Interval
from voice_pipeline_quality_filter_audio_part.repository import (
    ClaimDisposition,
    QualityFilterRepository,
)
from voice_pipeline_quality_filter_audio_part.storage import ObjectStorage
from voice_pipeline_quality_filter_audio_part.task import QualityFilterAudioPartHandler

pytestmark = pytest.mark.integration
if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Set RUN_INTEGRATION_TESTS=1 through tests/integration/run.sh.",
        allow_module_level=True,
    )


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing integration environment variable: {name}.")
    return value


def database_url(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def wav_bytes(duration_ms: int) -> bytes:
    samples = (np.sin(np.arange(duration_ms * 16) / 20) * 12000).astype("<i2")
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())
    return target.getvalue()


class FakeMusicDetector:
    model_name = "deterministic-integration-music"

    def __init__(self, intervals: tuple[Interval, ...] = ()) -> None:
        self._intervals = intervals

    def detect(self, _waveform, *, sample_rate: int, duration_ms: int):
        assert sample_rate == 16000
        assert duration_ms > 0
        return self._intervals


QUALITY = QualityPolicy(
    min_snr_db=10.0,
    music_probability_threshold=0.2,
    min_music_interval_ms=2000,
    music_gap_fill_ms=600,
    max_music_overlap_ratio=0.3,
    max_absorbable_bad_group_ms=3000,
    min_good_region_ms=20000,
)
PLANNER = PlannerPolicy(
    min_planning_window_ms=20000,
    max_planning_window_ms=60000,
    min_speaker_turn_ms=4000,
    min_speaker_total_ms=8000,
    backchannel_threshold_ms=1500,
    max_monologue_ms=40000,
    max_inner_iterations=200,
)
TASK = TaskPolicy(
    error_max_length=512,
    workspace_prefix="quality-integration-",
    max_diarization_bytes=1024 * 1024,
)


def settings(tmp_path: Path) -> EnvironmentSettings:
    return EnvironmentSettings(
        database_url=required("TEST_DATABASE_URL"),
        celery_broker_url=required("TEST_CELERY_BROKER_URL"),
        s3_bucket=required("TEST_S3_BUCKET"),
        s3_region=required("TEST_S3_REGION"),
        s3_endpoint_url=required("TEST_S3_ENDPOINT_URL"),
        music_model_cache_dir=tmp_path / "unused-model-cache",
    )


def put_part(
    connection,
    s3,
    environment: EnvironmentSettings,
    *,
    raw_audio_id: UUID,
    audio_part_id: UUID,
    part_index: int,
    duration_ms: int,
    turns: list[RawTurn],
) -> tuple[str, str]:
    prefix = f"integration/raw_audios/{raw_audio_id}/audio_parts/{part_index}"
    audio_key = f"{prefix}/audio.wav"
    diarization_key = f"{prefix}/diarization.json"
    s3.put_object(
        Bucket=environment.s3_bucket,
        Key=audio_key,
        Body=wav_bytes(duration_ms),
        ContentType="audio/wav",
    )
    s3.put_object(
        Bucket=environment.s3_bucket,
        Key=diarization_key,
        Body=build_artifact(
            turns, model="deterministic-integration-diarization", duration_ms=duration_ms
        ).to_json_bytes(),
        ContentType="application/json",
    )
    connection.execute(
        """
        INSERT INTO audio_parts
            (id, raw_audio_id, part_index, status, audio_uri, diarization_uri,
             relative_start_ms, relative_end_ms, duration_ms)
        VALUES (%s, %s, %s, 'diarized', %s, %s, 0, %s, %s)
        """,
        (
            audio_part_id,
            raw_audio_id,
            part_index,
            f"s3://{environment.s3_bucket}/{audio_key}",
            f"s3://{environment.s3_bucket}/{diarization_key}",
            duration_ms,
            duration_ms,
        ),
    )
    return audio_key, diarization_key


def test_real_services_chunks_zero_result_and_concurrent_claim(
    tmp_path: Path, monkeypatch
) -> None:
    load_dotenv(Path(__file__).parents[4] / ".env.test", override=True)
    os.environ["AWS_ACCESS_KEY_ID"] = required("TEST_AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = required("TEST_AWS_SECRET_ACCESS_KEY")
    environment = settings(tmp_path)
    s3 = boto3.client(
        "s3",
        endpoint_url=environment.s3_endpoint_url,
        region_name=environment.s3_region,
    )
    raw_audio_id = uuid4()
    accepted_id = uuid4()
    empty_id = uuid4()
    concurrent_id = uuid4()
    conflict_id = uuid4()
    conflict_chunk_id = uuid4()
    missing_artifact_id = uuid4()
    object_keys: list[str] = []
    with psycopg.connect(database_url(environment.database_url)) as connection:
        connection.execute(
            "INSERT INTO raw_audios (id, status, audio_uri) VALUES (%s, 'split_completed', %s)",
            (raw_audio_id, f"s3://{environment.s3_bucket}/integration/raw.wav"),
        )
        object_keys.extend(
            put_part(
                connection,
                s3,
                environment,
                raw_audio_id=raw_audio_id,
                audio_part_id=accepted_id,
                part_index=0,
                duration_ms=98000,
                turns=[
                    RawTurn(0, 10, "a"),
                    RawTurn(11, 21, "b"),
                    RawTurn(22, 32, "a"),
                    RawTurn(33, 46, "b"),
                    RawTurn(52, 62, "a"),
                    RawTurn(63, 73, "b"),
                    RawTurn(74, 84, "a"),
                    RawTurn(85, 98, "b"),
                ],
            )
        )
        object_keys.extend(
            put_part(
                connection,
                s3,
                environment,
                raw_audio_id=raw_audio_id,
                audio_part_id=empty_id,
                part_index=1,
                duration_ms=20000,
                turns=[],
            )
        )
        object_keys.extend(
            put_part(
                connection,
                s3,
                environment,
                raw_audio_id=raw_audio_id,
                audio_part_id=conflict_id,
                part_index=3,
                duration_ms=46000,
                turns=[
                    RawTurn(0, 10, "a"),
                    RawTurn(11, 21, "b"),
                    RawTurn(22, 32, "a"),
                    RawTurn(33, 46, "b"),
                ],
            )
        )
        connection.execute(
            """
            INSERT INTO chunks
                (id, audio_part_id, chunk_index, status, audio_uri, lang,
                 duration_ms, relative_start_ms, relative_end_ms)
            VALUES (%s, %s, 0, 'pending', %s, 'en', 46000, 0, 46000)
            """,
            (
                conflict_chunk_id,
                conflict_id,
                f"s3://{environment.s3_bucket}/integration/conflicting.wav",
            ),
        )
        connection.execute(
            """
            INSERT INTO audio_parts
                (id, raw_audio_id, part_index, status, audio_uri, diarization_uri,
                 relative_start_ms, relative_end_ms, duration_ms)
            VALUES (%s, %s, 4, 'diarized', %s, %s, 0, 98000, 98000)
            """,
            (
                missing_artifact_id,
                raw_audio_id,
                f"s3://{environment.s3_bucket}/integration/raw_audios/{raw_audio_id}"
                "/audio_parts/0/audio.wav",
                f"s3://{environment.s3_bucket}/integration/missing-diarization.json",
            ),
        )
        object_keys.extend(
            put_part(
                connection,
                s3,
                environment,
                raw_audio_id=raw_audio_id,
                audio_part_id=concurrent_id,
                part_index=2,
                duration_ms=20000,
                turns=[],
            )
        )

    monkeypatch.setattr(
        "voice_pipeline_quality_filter_audio_part.task.wada_snr", lambda _samples: 20.0
    )
    repository = QualityFilterRepository.create(environment)
    storage = ObjectStorage.create(environment, TASK)
    broker = Celery("quality-filter-integration", broker=environment.celery_broker_url)
    try:
        broker.send_task(
            QUALITY_FILTER_AUDIO_PART.name,
            args=[str(accepted_id)],
            queue=QUALITY_FILTER_AUDIO_PART.queue,
        )
        with (
            broker.connection_for_read() as connection,
            connection.SimpleQueue(QUALITY_FILTER_AUDIO_PART.queue) as queue,
        ):
            message = queue.get(block=True, timeout=5)
            try:
                assert message.payload[0] == [str(accepted_id)]
                assert message.payload[1] == {}
            finally:
                message.ack()

        handler = QualityFilterAudioPartHandler(
            repository=repository,
            storage=storage,
            music_detector=FakeMusicDetector((Interval(47000, 51000),)),
            quality_policy=QUALITY,
            planner_policy=PLANNER,
            task_policy=TASK,
            workspace_parent=tmp_path,
        )
        assert handler(str(accepted_id))["created_count"] == 2
        assert handler(str(accepted_id))["outcome"] == "already_completed"
        assert handler(str(empty_id))["created_count"] == 0

        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = tuple(
                pool.map(lambda _index: repository.claim(concurrent_id), range(2))
            )
        assert sorted(claim.disposition for claim in claims) == [
            ClaimDisposition.ALREADY_PROCESSING,
            ClaimDisposition.CLAIMED,
        ]
        assert repository.claim(concurrent_id).disposition is ClaimDisposition.ALREADY_PROCESSING
        with psycopg.connect(database_url(environment.database_url)) as connection:
            connection.execute(
                "UPDATE audio_parts SET status = 'failed' WHERE id = %s",
                (concurrent_id,),
            )
        recovered = repository.claim(concurrent_id)
        assert recovered.disposition is ClaimDisposition.CLAIMED
        repository.mark_failed(concurrent_id, "explicit recovery test")

        with pytest.raises(TaskStageError) as conflict_failure:
            handler(str(conflict_id))
        assert conflict_failure.value.code is ErrorCode.PERSISTENCE_FAILED
        expected_conflict_uri = (
            f"s3://{environment.s3_bucket}/integration/raw_audios/{raw_audio_id}"
            "/audio_parts/3/chunks/0/audio.wav"
        )
        with psycopg.connect(database_url(environment.database_url)) as connection:
            conflict_state = connection.execute(
                "SELECT status FROM audio_parts WHERE id = %s", (conflict_id,)
            ).fetchone()[0]
            conflict_rows = connection.execute(
                "SELECT id, audio_uri FROM chunks WHERE audio_part_id = %s",
                (conflict_id,),
            ).fetchall()
            connection.execute(
                "UPDATE chunks SET audio_uri = %s WHERE id = %s",
                (expected_conflict_uri, conflict_chunk_id),
            )
        assert conflict_state == "failed"
        assert conflict_rows == [
            (
                conflict_chunk_id,
                f"s3://{environment.s3_bucket}/integration/conflicting.wav",
            )
        ]
        assert handler(str(conflict_id))["created_count"] == 1
        object_keys.append(
            expected_conflict_uri.removeprefix(f"s3://{environment.s3_bucket}/")
        )
        with pytest.raises(TaskStageError) as missing_failure:
            handler(str(missing_artifact_id))
        assert missing_failure.value.code is ErrorCode.DOWNLOAD_FAILED

        with psycopg.connect(database_url(environment.database_url)) as connection:
            accepted = connection.execute(
                "SELECT status, error FROM audio_parts WHERE id = %s", (accepted_id,)
            ).fetchone()
            chunks = connection.execute(
                """
                SELECT status, audio_uri, duration_ms, relative_start_ms, relative_end_ms
                FROM chunks WHERE audio_part_id = %s
                ORDER BY chunk_index
                """,
                (accepted_id,),
            ).fetchall()
            empty = connection.execute(
                "SELECT status, error FROM audio_parts WHERE id = %s", (empty_id,)
            ).fetchone()
            empty_count = connection.execute(
                "SELECT count(*) FROM chunks WHERE audio_part_id = %s", (empty_id,)
            ).fetchone()[0]
            converged = connection.execute(
                "SELECT status FROM audio_parts WHERE id = %s", (conflict_id,)
            ).fetchone()[0]
            converged_chunk_id = connection.execute(
                "SELECT id FROM chunks WHERE audio_part_id = %s", (conflict_id,)
            ).fetchone()[0]
            missing_state = connection.execute(
                "SELECT status FROM audio_parts WHERE id = %s", (missing_artifact_id,)
            ).fetchone()[0]
        assert accepted == ("completed", None)
        assert len(chunks) == 2
        assert all(chunk[0] == "pending" for chunk in chunks)
        assert [chunk[1].rsplit("/", 2)[-2] for chunk in chunks] == ["0", "1"]
        assert all(chunk[2] == chunk[4] - chunk[3] > 0 for chunk in chunks)
        assert empty == ("completed", None)
        assert empty_count == 0
        assert converged == "completed"
        assert converged_chunk_id == conflict_chunk_id
        assert missing_state == "failed"
        for chunk in chunks:
            chunk_key = chunk[1].removeprefix(f"s3://{environment.s3_bucket}/")
            object_keys.append(chunk_key)
            response = s3.head_object(Bucket=environment.s3_bucket, Key=chunk_key)
            assert response["ContentType"] == "audio/wav"
            assert response["ContentLength"] > 0
    finally:
        broker.close()
        storage.close()
        repository.close()
        with psycopg.connect(database_url(environment.database_url)) as connection:
            connection.execute("DELETE FROM raw_audios WHERE id = %s", (raw_audio_id,))
        if object_keys:
            s3.delete_objects(
                Bucket=environment.s3_bucket,
                Delete={"Objects": [{"Key": key} for key in object_keys]},
            )
        s3.close()
