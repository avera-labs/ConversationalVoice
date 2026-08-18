from __future__ import annotations

from uuid import UUID

import boto3

from voice_pipeline_ingest_api.config import EnvironmentSettings
from voice_pipeline_ingest_api.services.storage import ObjectStorage


class FakeS3Client:
    def __init__(self) -> None:
        self.upload_calls: list[tuple] = []
        self.delete_calls: list[dict] = []

    def upload_file(self, *args, **kwargs) -> None:
        self.upload_calls.append((*args, kwargs))

    def delete_object(self, **kwargs) -> None:
        self.delete_calls.append(kwargs)


def test_upload_normalized_audio_uses_deterministic_key_and_content_type(
    tmp_path,
) -> None:
    client = FakeS3Client()
    storage = ObjectStorage(client=client, bucket="test-bucket")
    raw_audio_id = UUID("12345678-1234-5678-1234-567812345678")
    path = tmp_path / "audio.wav"
    path.write_bytes(b"wav")

    uri = storage.upload_normalized_audio(raw_audio_id, path)

    key = "raw_audios/12345678-1234-5678-1234-567812345678/audio.wav"
    assert client.upload_calls == [
        (str(path), "test-bucket", key, {"ExtraArgs": {"ContentType": "audio/wav"}})
    ]
    assert uri == f"s3://test-bucket/{key}"


def test_delete_normalized_audio_uses_same_deterministic_key() -> None:
    client = FakeS3Client()
    storage = ObjectStorage(client=client, bucket="test-bucket")
    raw_audio_id = UUID("12345678-1234-5678-1234-567812345678")

    storage.delete_normalized_audio(raw_audio_id)

    assert client.delete_calls == [
        {
            "Bucket": "test-bucket",
            "Key": "raw_audios/12345678-1234-5678-1234-567812345678/audio.wav",
        }
    ]


def test_aws_sdk_resolves_static_credentials_from_standard_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")

    credentials = boto3.Session().get_credentials()

    assert credentials is not None
    frozen = credentials.get_frozen_credentials()
    assert frozen.access_key == "test-access-key"
    assert frozen.secret_key == "test-secret-key"


def test_object_storage_does_not_require_explicit_credential_options(
    monkeypatch,
) -> None:
    captured: dict = {}

    def create_client(service_name: str, **options):
        captured["service_name"] = service_name
        captured["options"] = options
        return FakeS3Client()

    monkeypatch.setattr(
        "voice_pipeline_ingest_api.services.storage.boto3.client",
        create_client,
    )
    settings = EnvironmentSettings(
        database_url="postgresql+psycopg://user:password@db/voice",
        celery_broker_url="redis://redis:6379/0",
        s3_bucket="test-bucket",
        s3_region="us-east-1",
    )

    storage = ObjectStorage.create(settings)

    assert storage.bucket == "test-bucket"
    assert captured == {
        "service_name": "s3",
        "options": {"region_name": "us-east-1"},
    }
