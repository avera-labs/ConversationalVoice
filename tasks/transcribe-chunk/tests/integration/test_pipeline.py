import hashlib
import io
import os
import wave
from pathlib import Path
from uuid import uuid4

import boto3
import numpy as np
import psycopg
import pytest
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from voice_pipeline_transcribe_chunk.config import load_settings
from voice_pipeline_transcribe_chunk.repository import Repository
from voice_pipeline_transcribe_chunk.storage import ObjectStorage
from voice_pipeline_transcribe_chunk.task import Handler
from voice_pipeline_transcribe_chunk.utterances import DecodedWord

pytestmark = pytest.mark.integration
if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Set RUN_INTEGRATION_TESTS=1 through tests/integration/run.sh.",
        allow_module_level=True,
    )


def required(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing integration environment variable: {name}.")
    return value


def wav_bytes(duration_ms=4000):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.zeros(duration_ms * 16, dtype="<i2").tobytes())
    return target.getvalue()


class Model:
    def transcribe(self, _audio):
        return [DecodedWord("Hello.", 0.1, 0.4, 0.9)]


def test_real_database_and_storage_pipeline(tmp_path):
    load_dotenv(Path(__file__).parents[4] / ".env.test", override=True)
    environment = {
        "DATABASE_URL": required("TEST_DATABASE_URL"),
        "CELERY_BROKER_URL": required("TEST_CELERY_BROKER_URL"),
        "S3_BUCKET": required("TEST_S3_BUCKET"),
        "S3_REGION": required("TEST_S3_REGION"),
        "S3_ENDPOINT_URL": required("TEST_S3_ENDPOINT_URL"),
    }
    settings = load_settings(environment)
    os.environ["AWS_ACCESS_KEY_ID"] = required("TEST_AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = required("TEST_AWS_SECRET_ACCESS_KEY")
    s3 = boto3.client(
        "s3",
        endpoint_url=environment["S3_ENDPOINT_URL"],
        region_name=environment["S3_REGION"],
    )
    raw_id, part_id, chunk_id = uuid4(), uuid4(), uuid4()
    prefix = f"integration/raw_audios/{raw_id}/audio_parts/0/chunks/0"
    audio_uri = f"s3://{environment['S3_BUCKET']}/{prefix}/audio.wav"
    speaker_data = wav_bytes()
    speaker_sha = hashlib.sha256(speaker_data).hexdigest()
    speaker_uris = tuple(
        f"s3://{environment['S3_BUCKET']}/{prefix}/results/separated/speaker-{slot}.wav"
        for slot in range(2)
    )
    for uri in speaker_uris:
        s3.put_object(
            Bucket=environment["S3_BUCKET"],
            Key=uri.split(f"s3://{environment['S3_BUCKET']}/", 1)[1],
            Body=speaker_data,
            ContentType="audio/wav",
        )
    snapshot = {
        "schema_version": 1,
        "timebase": "chunk",
        "segments": [
            {"speaker": 4, "start_ms": 100, "end_ms": 1000, "duration_ms": 900},
            {"speaker": 7, "start_ms": 2000, "end_ms": 3000, "duration_ms": 1000},
        ],
    }
    separation = {
        "schema_version": 1,
        "backend": "dialogue_sidon",
        "model": {
            "repo_id": "sarulab-speech/DialogueSidon",
            "revision": "a" * 40,
            "config_version": "sidon-v1",
            "inference_steps": 100,
        },
        "input_audio": {
            "sample_rate_hz": 16000,
            "duration_ms": 4000,
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
        "speaker_audio": [
            {
                "output_slot": slot,
                "diarization_speaker_id": speaker,
                "uri": speaker_uris[slot],
                "sample_rate_hz": 16000,
                "duration_ms": 4000,
                "size_bytes": len(speaker_data),
                "sha256": speaker_sha,
            }
            for slot, speaker in enumerate((4, 7))
        ],
        "audit": {
            "verdict": "ok",
            "reference_speaker_id": 4,
            "consistent_relation": "direct",
        },
    }
    database_url = environment["DATABASE_URL"].replace(
        "postgresql+psycopg://", "postgresql://", 1
    )
    with psycopg.connect(database_url) as connection:
        connection.execute(
            "INSERT INTO raw_audios (id,status,audio_uri) VALUES (%s,'split_completed',%s)",
            (raw_id, f"s3://{environment['S3_BUCKET']}/integration/raw.wav"),
        )
        connection.execute(
            "INSERT INTO audio_parts (id,raw_audio_id,part_index,status,audio_uri,relative_start_ms,relative_end_ms,duration_ms) VALUES (%s,%s,0,'completed',%s,0,4000,4000)",
            (part_id, raw_id, f"s3://{environment['S3_BUCKET']}/integration/part.wav"),
        )
        connection.execute(
            "INSERT INTO chunks (id,audio_part_id,chunk_index,status,audio_uri,duration_ms,relative_start_ms,relative_end_ms,diarizations,final_results) VALUES (%s,%s,0,'separated',%s,4000,0,4000,%s,%s)",
            (
                chunk_id,
                part_id,
                audio_uri,
                Jsonb(snapshot),
                Jsonb({"separation": separation}),
            ),
        )
    repository = Repository.create(settings.environment)
    storage = ObjectStorage.create(settings.environment)
    try:
        result = Handler(repository, storage, Model(), settings.policy, tmp_path)(
            str(chunk_id)
        )
        assert result["outcome"] == "transcribed"
        with psycopg.connect(database_url) as connection:
            row = connection.execute(
                "SELECT status,final_results FROM chunks WHERE id=%s", (chunk_id,)
            ).fetchone()
            assert row[0] == "transcribed"
            assert [
                item["diarization_speaker_id"]
                for item in row[1]["transcription"]["input_speaker_audio"]
            ] == [4, 7]
            connection.execute("DELETE FROM raw_audios WHERE id=%s", (raw_id,))
    finally:
        for key in (
            *(
                uri.split(f"s3://{environment['S3_BUCKET']}/", 1)[1]
                for uri in speaker_uris
            ),
            f"{prefix}/results/transcript.json",
            f"{prefix}/results/word_alignment.json",
        ):
            s3.delete_object(Bucket=environment["S3_BUCKET"], Key=key)
        storage.close()
        repository.close()
