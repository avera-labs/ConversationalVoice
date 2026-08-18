from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from voice_pipeline_split_raw_audio_into_parts.storage import (
    InvalidS3UriError,
    ObjectStorage,
    StorageError,
    parse_s3_uri,
)


RAW_AUDIO_ID = UUID("12345678-1234-5678-1234-567812345678")


class FakeS3Client:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str, str]] = []
        self.uploads: list[tuple[str, str, str, dict[str, str]]] = []
        self.download_error: Exception | None = None
        self.upload_error: Exception | None = None
        self.closed = False

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        self.downloads.append((bucket, key, destination))
        Path(destination).write_bytes(b"partial-or-complete")
        if self.download_error is not None:
            raise self.download_error

    def upload_file(
        self,
        source: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict[str, str],
    ) -> None:
        self.uploads.append((source, bucket, key, ExtraArgs))
        if self.upload_error is not None:
            raise self.upload_error

    def head_bucket(self, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_parse_s3_uri_accepts_only_configured_bucket() -> None:
    location = parse_s3_uri(
        "s3://test-bucket/raw_audios/id/audio.wav",
        expected_bucket="test-bucket",
    )

    assert location.bucket == "test-bucket"
    assert location.key == "raw_audios/id/audio.wav"


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "https://test-bucket/raw.wav",
        "s3://other-bucket/raw.wav",
        "s3://test-bucket",
        "s3://test-bucket//raw.wav",
        "s3://test-bucket/a/../raw.wav",
        "s3://test-bucket/raw%2Ffile.wav",
        "s3://test-bucket/raw.wav?signature=secret",
        "s3://test-bucket/raw.wav#fragment",
        "s3://test-bucket/raw file.wav",
    ],
)
def test_parse_s3_uri_rejects_ambiguous_or_external_uri(uri: str) -> None:
    with pytest.raises(InvalidS3UriError, match="URI is invalid"):
        parse_s3_uri(uri, expected_bucket="test-bucket")


def test_download_uses_validated_location(tmp_path: Path) -> None:
    client = FakeS3Client()
    storage = ObjectStorage(client, "test-bucket")
    destination = tmp_path / "raw.wav"

    storage.download_raw_audio(
        "s3://test-bucket/raw_audios/id/audio.wav",
        destination,
    )

    assert destination.read_bytes() == b"partial-or-complete"
    assert client.downloads == [
        ("test-bucket", "raw_audios/id/audio.wav", str(destination))
    ]


def test_failed_download_removes_partial_file_and_hides_exception(
    tmp_path: Path,
) -> None:
    secret = "signed-url-secret"
    client = FakeS3Client()
    client.download_error = RuntimeError(secret)
    storage = ObjectStorage(client, "test-bucket")
    destination = tmp_path / "raw.wav"

    with pytest.raises(StorageError) as error:
        storage.download_raw_audio(
            "s3://test-bucket/raw_audios/id/audio.wav",
            destination,
        )

    assert str(error.value) == "Unable to download the normalized WAV."
    assert secret not in str(error.value)
    assert not destination.exists()


def test_uploads_use_deterministic_keys_and_content_types(tmp_path: Path) -> None:
    client = FakeS3Client()
    storage = ObjectStorage(client, "test-bucket")
    vad_path = tmp_path / "vad_segments.json"
    part_path = tmp_path / "audio.wav"
    vad_path.write_text('{"segments": []}', encoding="utf-8")
    part_path.write_bytes(b"RIFF-test")

    vad_uri = storage.upload_vad_segments(RAW_AUDIO_ID, vad_path)
    part_uri = storage.upload_audio_part(RAW_AUDIO_ID, 2, part_path)

    assert vad_uri == (
        "s3://test-bucket/raw_audios/"
        f"{RAW_AUDIO_ID}/vad_segments.json"
    )
    assert part_uri == (
        "s3://test-bucket/raw_audios/"
        f"{RAW_AUDIO_ID}/audio_parts/2/audio.wav"
    )
    assert client.uploads[0][2:] == (
        f"raw_audios/{RAW_AUDIO_ID}/vad_segments.json",
        {"ContentType": "application/json"},
    )
    assert client.uploads[1][2:] == (
        f"raw_audios/{RAW_AUDIO_ID}/audio_parts/2/audio.wav",
        {"ContentType": "audio/wav"},
    )


def test_upload_failure_is_safe_and_never_deletes_existing_object(
    tmp_path: Path,
) -> None:
    secret = "private-endpoint-response"
    client = FakeS3Client()
    client.upload_error = RuntimeError(secret)
    storage = ObjectStorage(client, "test-bucket")
    part_path = tmp_path / "audio.wav"
    part_path.write_bytes(b"RIFF-test")

    with pytest.raises(StorageError) as error:
        storage.upload_audio_part(RAW_AUDIO_ID, 0, part_path)

    assert str(error.value) == "Unable to upload an audio part."
    assert secret not in str(error.value)
    assert not hasattr(client, "delete_object")


def test_empty_local_artifact_is_rejected_before_upload(tmp_path: Path) -> None:
    client = FakeS3Client()
    storage = ObjectStorage(client, "test-bucket")
    empty = tmp_path / "empty.wav"
    empty.touch()

    with pytest.raises(StorageError, match="Unable to upload an audio part"):
        storage.upload_audio_part(RAW_AUDIO_ID, 0, empty)

    assert client.uploads == []


def test_storage_close_releases_sdk_client() -> None:
    client = FakeS3Client()
    storage = ObjectStorage(client, "test-bucket")

    storage.close()

    assert client.closed is True
