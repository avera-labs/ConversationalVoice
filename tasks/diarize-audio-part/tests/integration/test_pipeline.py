from __future__ import annotations

import hashlib
import io
import json
import os
import wave
from pathlib import Path
from uuid import uuid4

import boto3
import psycopg
import pytest
from celery import Celery
from dotenv import load_dotenv
from voice_pipeline_task_contracts import QUALITY_FILTER_AUDIO_PART

from voice_pipeline_diarize_audio_part.artifact import RawTurn
from voice_pipeline_diarize_audio_part.config import (
    DiarizationPolicy,
    EnvironmentSettings,
    SpeakerReferencePolicy,
    TaskPolicy,
)
from voice_pipeline_diarize_audio_part.diarization import InferenceResult
from voice_pipeline_diarize_audio_part.publisher import QualityFilterPublisher
from voice_pipeline_diarize_audio_part.repository import DiarizationRepository
from voice_pipeline_diarize_audio_part.storage import ObjectStorage
from voice_pipeline_diarize_audio_part.task import DiarizeAudioPartHandler

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


class FakeDiarization:
    calls = 0

    def infer(self, audio_path: Path) -> InferenceResult:
        assert audio_path.stat().st_size > 0
        self.calls += 1
        return InferenceResult(
            turns=(RawTurn(0.0, 7.0, "first"), RawTurn(7.0, 8.0, "second")),
            device="cpu",
            accelerator=None,
            model_cache_hit=self.calls > 1,
        )


def wav_bytes() -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(bytes(16000 * 8 * 2))
    return target.getvalue()


def test_real_services_happy_path_and_message_contract(tmp_path: Path) -> None:
    load_dotenv(Path(__file__).parents[4] / ".env.test", override=True)
    os.environ["AWS_ACCESS_KEY_ID"] = required("TEST_AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = required("TEST_AWS_SECRET_ACCESS_KEY")
    settings = EnvironmentSettings(
        database_url=required("TEST_DATABASE_URL"),
        celery_broker_url=required("TEST_CELERY_BROKER_URL"),
        s3_bucket=required("TEST_S3_BUCKET"),
        s3_region=required("TEST_S3_REGION"),
        s3_endpoint_url=required("TEST_S3_ENDPOINT_URL"),
    )
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
    )
    raw_audio_id = uuid4()
    audio_part_id = uuid4()
    part_prefix = f"raw_audios/{raw_audio_id}/audio_parts/0"
    input_key = f"{part_prefix}/audio.wav"
    artifact_key = f"{part_prefix}/diarization.json"
    reference_manifest_key = f"{part_prefix}/speaker-references/references.json"
    reference_audio_key = f"{part_prefix}/speaker-references/speaker-0.wav"
    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=input_key,
        Body=wav_bytes(),
        ContentType="audio/wav",
    )
    with psycopg.connect(database_url(settings.database_url)) as connection:
        connection.execute(
            "INSERT INTO raw_audios (id, status, audio_uri) VALUES (%s, 'split_completed', %s)",
            (raw_audio_id, f"s3://{settings.s3_bucket}/integration/raw.wav"),
        )
        connection.execute(
            """
            INSERT INTO audio_parts
                (id, raw_audio_id, part_index, status, audio_uri,
                relative_start_ms, relative_end_ms, duration_ms)
            VALUES (%s, %s, 0, 'pending', %s, 0, 8000, 8000)
            """,
            (audio_part_id, raw_audio_id, f"s3://{settings.s3_bucket}/{input_key}"),
        )

    repository = DiarizationRepository.create(settings)
    storage = ObjectStorage.create(settings)
    publisher = QualityFilterPublisher.create(settings)
    broker = Celery("diarization-integration-reader", broker=settings.celery_broker_url)
    try:
        handler = DiarizeAudioPartHandler(
            repository=repository,
            storage=storage,
            diarization=FakeDiarization(),
            publisher=publisher,
            diarization_policy=DiarizationPolicy(
                model="deterministic-integration-model",
                device="cpu",
            ),
            speaker_reference_policy=SpeakerReferencePolicy(
                min_segment_ms=4000,
                edge_trim_ms=500,
                min_speaker_effective_ms=4000,
                max_speaker_effective_ms=30000,
                inter_segment_silence_ms=500,
            ),
            task_policy=TaskPolicy(
                error_max_length=512, workspace_prefix="integration-diarization-"
            ),
            workspace_parent=tmp_path,
        )
        result = handler(str(audio_part_id))
        assert result["status"] == "diarized"

        with psycopg.connect(database_url(settings.database_url)) as connection:
            row = connection.execute(
                "SELECT status, diarization_uri, error FROM audio_parts WHERE id = %s",
                (audio_part_id,),
            ).fetchone()
        assert row == ("diarized", f"s3://{settings.s3_bucket}/{artifact_key}", None)
        response = s3.get_object(Bucket=settings.s3_bucket, Key=artifact_key)
        assert response["ContentType"] == "application/json"
        artifact = json.loads(response["Body"].read())
        assert artifact["schema_version"] == 1
        assert artifact["num_speakers"] == 2
        response = s3.get_object(Bucket=settings.s3_bucket, Key=reference_manifest_key)
        assert response["ContentType"] == "application/json"
        manifest = json.loads(response["Body"].read())
        assert manifest["schema_version"] == 1
        assert len(manifest["speakers"]) == 1
        reference = manifest["speakers"][0]
        assert reference["speaker_id"] == 0
        assert reference["reference_audio"]["segments"] == [
            {"start_ms": 500, "end_ms": 6500, "duration_ms": 6000}
        ]
        assert reference["reference_audio"]["effective_duration_ms"] == 6000
        assert reference["reference_audio"]["total_duration_ms"] == 6000
        response = s3.get_object(Bucket=settings.s3_bucket, Key=reference_audio_key)
        assert response["ContentType"] == "audio/wav"
        reference_bytes = response["Body"].read()
        assert reference["reference_audio"]["size_bytes"] == len(reference_bytes)
        assert (
            reference["reference_audio"]["sha256"]
            == hashlib.sha256(reference_bytes).hexdigest()
        )

        with (
            broker.connection_for_read() as connection,
            connection.SimpleQueue(QUALITY_FILTER_AUDIO_PART.queue) as queue,
        ):
            message = queue.get(block=True, timeout=5)
            try:
                assert message.payload[0] == [str(audio_part_id)]
                assert message.payload[1] == {}
            finally:
                message.ack()
    finally:
        broker.close()
        publisher.close()
        storage.close()
        repository.close()
        with psycopg.connect(database_url(settings.database_url)) as connection:
            connection.execute("DELETE FROM raw_audios WHERE id = %s", (raw_audio_id,))
        s3.delete_objects(
            Bucket=settings.s3_bucket,
            Delete={
                "Objects": [
                    {"Key": input_key},
                    {"Key": artifact_key},
                    {"Key": reference_manifest_key},
                    {"Key": reference_audio_key},
                ]
            },
        )
        s3.close()
