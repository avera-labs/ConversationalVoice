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
from voice_pipeline_diarization_artifact import RawTurn, build_artifact

from voice_pipeline_separate_chunk.config import load_settings
from voice_pipeline_separate_chunk.repository import Repository
from voice_pipeline_separate_chunk.storage import ObjectStorage
from voice_pipeline_separate_chunk.task import Handler

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


def wav_bytes():
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.zeros(20000 * 16, dtype="<i2").tobytes())
    return target.getvalue()


class Model:
    def separate(self, samples, seed):
        tracks = np.zeros((2, len(samples)), dtype=np.float32)
        tracks[0, : 9000 * 16] = 0.5
        tracks[1, 10000 * 16 : 19000 * 16] = 0.5
        return tracks, 16000


class Aligner:
    def align(self, *args):
        raise AssertionError("single window must not align")


def test_real_database_and_storage_pipeline(tmp_path):
    load_dotenv(Path(__file__).parents[4] / ".env.test", override=True)
    env = {
        "DATABASE_URL": required("TEST_DATABASE_URL"),
        "CELERY_BROKER_URL": required("TEST_CELERY_BROKER_URL"),
        "S3_BUCKET": required("TEST_S3_BUCKET"),
        "S3_REGION": required("TEST_S3_REGION"),
        "S3_ENDPOINT_URL": required("TEST_S3_ENDPOINT_URL"),
    }
    settings = load_settings(env)
    os.environ["AWS_ACCESS_KEY_ID"] = required("TEST_AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = required("TEST_AWS_SECRET_ACCESS_KEY")
    s3 = boto3.client(
        "s3", endpoint_url=env["S3_ENDPOINT_URL"], region_name=env["S3_REGION"]
    )
    raw_id, part_id, chunk_id = uuid4(), uuid4(), uuid4()
    prefix = f"integration/raw_audios/{raw_id}/audio_parts/0"
    audio_key = f"{prefix}/chunks/0/audio.wav"
    diar_key = f"{prefix}/diarization.json"
    s3.put_object(
        Bucket=env["S3_BUCKET"],
        Key=audio_key,
        Body=wav_bytes(),
        ContentType="audio/wav",
    )
    s3.put_object(
        Bucket=env["S3_BUCKET"],
        Key=diar_key,
        Body=build_artifact(
            (RawTurn(0, 9, "a"), RawTurn(10, 19, "b")),
            model="integration",
            duration_ms=20000,
        ).to_json_bytes(),
        ContentType="application/json",
    )
    database = env["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(database) as connection:
        connection.execute(
            "INSERT INTO raw_audios (id,status,audio_uri) VALUES (%s,'split_completed',%s)",
            (raw_id, f"s3://{env['S3_BUCKET']}/integration/raw.wav"),
        )
        connection.execute(
            "INSERT INTO audio_parts (id,raw_audio_id,part_index,status,audio_uri,diarization_uri,relative_start_ms,relative_end_ms,duration_ms) VALUES (%s,%s,0,'completed',%s,%s,0,20000,20000)",
            (
                part_id,
                raw_id,
                f"s3://{env['S3_BUCKET']}/{prefix}/audio.wav",
                f"s3://{env['S3_BUCKET']}/{diar_key}",
            ),
        )
        connection.execute(
            "INSERT INTO chunks (id,audio_part_id,chunk_index,status,audio_uri,duration_ms,relative_start_ms,relative_end_ms) VALUES (%s,%s,0,'pending',%s,20000,0,20000)",
            (chunk_id, part_id, f"s3://{env['S3_BUCKET']}/{audio_key}"),
        )
    repository = Repository.create(settings.environment)
    storage = ObjectStorage.create(settings.environment)
    try:
        result = Handler(
            repository, storage, Model(), Aligner(), settings.policy, tmp_path
        )(str(chunk_id))
        assert result["outcome"] == "separated"
        with psycopg.connect(database) as connection:
            row = connection.execute(
                "SELECT status,final_results FROM chunks WHERE id=%s", (chunk_id,)
            ).fetchone()
            assert row[0] == "separated"
            assert [
                x["diarization_speaker_id"]
                for x in row[1]["separation"]["speaker_audio"]
            ] == [0, 1]
            connection.execute("DELETE FROM raw_audios WHERE id=%s", (raw_id,))
        for key in (
            audio_key,
            diar_key,
            f"{prefix}/chunks/0/results/separated/speaker-0.wav",
            f"{prefix}/chunks/0/results/separated/speaker-1.wav",
        ):
            s3.delete_object(Bucket=env["S3_BUCKET"], Key=key)
    finally:
        storage.close()
        repository.close()
