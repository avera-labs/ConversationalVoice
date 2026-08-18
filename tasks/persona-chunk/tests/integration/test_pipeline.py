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
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from voice_pipeline_persona_chunk.config import load_settings
from voice_pipeline_persona_chunk.repository import Repository
from voice_pipeline_persona_chunk.storage import ObjectStorage
from voice_pipeline_persona_chunk.task import Handler

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


def wav_bytes(duration_ms=1000):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * (duration_ms * 16))
    return target.getvalue()


def transcript_document():
    model = {
        "repo_id": "nvidia/parakeet-tdt-0.6b-v3",
        "revision": "b" * 40,
        "config_version": "parakeet-v1",
    }
    return {
        "schema_version": 1,
        "backend": "parakeet_tdt",
        "model": model,
        "language": "en",
        "timebase": "chunk",
        "speakers": [
            {
                "output_slot": 0,
                "diarization_speaker_id": 4,
                "utterances": [
                    {
                        "utterance_index": 0,
                        "start_ms": 0,
                        "end_ms": 400,
                        "text": "Hello",
                        "confidence": 0.9,
                    }
                ],
            },
            {
                "output_slot": 1,
                "diarization_speaker_id": 7,
                "utterances": [
                    {
                        "utterance_index": 0,
                        "start_ms": 500,
                        "end_ms": 900,
                        "text": "Hi",
                        "confidence": 0.9,
                    }
                ],
            },
        ],
    }


class Client:
    def analyze(self, _mp3, _srt, mapping):
        assert mapping == (4, 7)
        speaker = {
            "name": None,
            "age": None,
            "ethnicity": None,
            "gender": None,
            "tag": "You enjoy having a good conversation.",
            "alpha": "low",
            "evidence": None,
            "primary_emotion": "neutral",
            "secondary_emotion": None,
            "emotion_intensity": "low",
            "laugh": False,
            "cry": False,
            "whisper": False,
            "shout": False,
            "sigh": False,
            "overall_tone": "calm",
        }
        return (
            {
                "scene": {
                    "description": "A calm exchange.",
                    "overall_tone": "calm",
                    "emotion_intensity": "low",
                },
                "speakers": {"4": dict(speaker), "7": dict(speaker)},
            },
            {
                "model": "xiaomi/mimo-v2.5",
                "in_tokens": 0,
                "out_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0,
            },
        )


class Publisher:
    def publish(self, _identifier):
        return "message-id"


def test_real_database_storage_and_ffmpeg_with_fake_provider(tmp_path):
    load_dotenv(Path(__file__).parents[4] / ".env.test", override=True)
    environment = {
        "DATABASE_URL": required("TEST_DATABASE_URL"),
        "CELERY_BROKER_URL": required("TEST_CELERY_BROKER_URL"),
        "S3_BUCKET": required("TEST_S3_BUCKET"),
        "S3_REGION": required("TEST_S3_REGION"),
        "S3_ENDPOINT_URL": required("TEST_S3_ENDPOINT_URL"),
        "OPENROUTER_API_KEY": "integration-placeholder",
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
    transcript_uri = f"s3://{environment['S3_BUCKET']}/{prefix}/results/transcript.json"
    persona_key = f"{prefix}/results/persona.json"
    audio = wav_bytes()
    audio_sha = hashlib.sha256(audio).hexdigest()
    transcript = json.dumps(
        transcript_document(), sort_keys=True, separators=(",", ":")
    ).encode()
    transcript_sha = hashlib.sha256(transcript).hexdigest()
    for key, body, content_type in (
        (f"{prefix}/audio.wav", audio, "audio/wav"),
        (f"{prefix}/results/transcript.json", transcript, "application/json"),
    ):
        s3.put_object(
            Bucket=environment["S3_BUCKET"],
            Key=key,
            Body=body,
            ContentType=content_type,
        )
    speaker_uris = tuple(
        f"s3://{environment['S3_BUCKET']}/{prefix}/results/separated/speaker-{slot}.wav"
        for slot in range(2)
    )
    snapshot = {
        "schema_version": 1,
        "timebase": "chunk",
        "segments": [
            {"speaker": 4, "start_ms": 0, "end_ms": 500, "duration_ms": 500},
            {"speaker": 7, "start_ms": 500, "end_ms": 1000, "duration_ms": 500},
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
            "duration_ms": 1000,
            "size_bytes": len(audio),
            "sha256": audio_sha,
        },
        "speaker_audio": [
            {
                "output_slot": slot,
                "diarization_speaker_id": speaker,
                "uri": speaker_uris[slot],
                "sample_rate_hz": 16000,
                "duration_ms": 1000,
                "size_bytes": 1,
                "sha256": str(slot + 1) * 64,
            }
            for slot, speaker in enumerate((4, 7))
        ],
        "audit": {
            "verdict": "ok",
            "reference_speaker_id": 4,
            "consistent_relation": "direct",
        },
    }
    transcription = {
        "schema_version": 1,
        "backend": "parakeet_tdt",
        "model": transcript_document()["model"],
        "language": "en",
        "input_speaker_audio": [
            {
                key: item[key]
                for key in (
                    "output_slot",
                    "diarization_speaker_id",
                    "uri",
                    "size_bytes",
                    "sha256",
                )
            }
            for item in separation["speaker_audio"]
        ],
        "artifacts": {
            "transcript": {
                "uri": transcript_uri,
                "size_bytes": len(transcript),
                "sha256": transcript_sha,
            },
            "word_alignment": {
                "uri": f"s3://{environment['S3_BUCKET']}/{prefix}/results/word_alignment.json",
                "size_bytes": 1,
                "sha256": "c" * 64,
            },
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
            "INSERT INTO audio_parts (id,raw_audio_id,part_index,status,audio_uri,relative_start_ms,relative_end_ms,duration_ms) VALUES (%s,%s,0,'completed',%s,0,1000,1000)",
            (part_id, raw_id, f"s3://{environment['S3_BUCKET']}/integration/part.wav"),
        )
        connection.execute(
            "INSERT INTO chunks (id,audio_part_id,chunk_index,status,audio_uri,duration_ms,relative_start_ms,relative_end_ms,diarizations,final_results) VALUES (%s,%s,0,'transcribed',%s,1000,0,1000,%s,%s)",
            (
                chunk_id,
                part_id,
                audio_uri,
                Jsonb(snapshot),
                Jsonb({"separation": separation, "transcription": transcription}),
            ),
        )
    repository = Repository.create(settings.environment)
    storage = ObjectStorage.create(settings.environment)
    try:
        result = Handler(
            repository, storage, Client(), Publisher(), settings.policy, tmp_path
        )(str(chunk_id))
        assert result["outcome"] == "persona_generated"
        with psycopg.connect(database_url) as connection:
            row = connection.execute(
                "SELECT status,persona,final_results FROM chunks WHERE id=%s",
                (chunk_id,),
            ).fetchone()
            assert row[0] == "persona_generated"
            assert row[1]["speaker_mapping"][0] == {
                "output_slot": 0,
                "diarization_speaker_id": 4,
            }
            assert row[2]["persona"]["artifact"]["uri"].endswith("persona.json")
            connection.execute("DELETE FROM raw_audios WHERE id=%s", (raw_id,))
    finally:
        for key in (
            f"{prefix}/audio.wav",
            f"{prefix}/results/transcript.json",
            persona_key,
        ):
            s3.delete_object(Bucket=environment["S3_BUCKET"], Key=key)
        storage.close()
        repository.close()
