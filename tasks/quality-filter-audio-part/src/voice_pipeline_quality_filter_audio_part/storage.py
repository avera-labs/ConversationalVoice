"""Canonical S3 input and deterministic chunk storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import boto3

from .config import EnvironmentSettings, TaskPolicy


class StorageError(RuntimeError):
    pass


class InvalidS3UriError(StorageError):
    pass


@dataclass(frozen=True, slots=True)
class S3Location:
    bucket: str
    key: str


def parse_s3_uri(uri: str, *, expected_bucket: str) -> S3Location:
    if not uri or any(character.isspace() for character in uri):
        raise InvalidS3UriError("Artifact URI is invalid.")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or parsed.netloc != expected_bucket
        or not parsed.path.startswith("/")
        or parsed.path.startswith("//")
        or parsed.query
        or parsed.fragment
        or "@" in parsed.netloc
        or ":" in parsed.netloc
        or "%" in parsed.path
        or "\\" in parsed.path
    ):
        raise InvalidS3UriError("Artifact URI is invalid.")
    key = parsed.path[1:]
    if not key or any(segment in {"", ".", ".."} for segment in key.split("/")):
        raise InvalidS3UriError("Artifact URI is invalid.")
    return S3Location(parsed.netloc, key)


class ObjectStorage:
    def __init__(self, client: Any, bucket: str, *, max_diarization_bytes: int) -> None:
        self._client = client
        self._bucket = bucket
        self._max_diarization_bytes = max_diarization_bytes

    @classmethod
    def create(cls, settings: EnvironmentSettings, policy: TaskPolicy) -> ObjectStorage:
        options: dict[str, str] = {"region_name": settings.s3_region}
        if settings.s3_endpoint_url is not None:
            options["endpoint_url"] = settings.s3_endpoint_url
        return cls(
            boto3.client("s3", **options),
            settings.s3_bucket,
            max_diarization_bytes=policy.max_diarization_bytes,
        )

    def _download(self, uri: str, destination: Path, *, maximum_bytes: int | None = None) -> int:
        location = parse_s3_uri(uri, expected_bucket=self._bucket)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(location.bucket, location.key, str(destination))
            size = destination.stat().st_size
            if size <= 0 or (maximum_bytes is not None and size > maximum_bytes):
                raise OSError("object size is invalid")
            return size
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise StorageError("Unable to download input artifact.") from exc

    def download_audio(self, uri: str, destination: Path) -> int:
        return self._download(uri, destination)

    def download_diarization(self, uri: str, destination: Path) -> int:
        return self._download(uri, destination, maximum_bytes=self._max_diarization_bytes)

    def chunk_key(self, audio_uri: str, chunk_index: int) -> str:
        if chunk_index < 0:
            raise ValueError("chunk index must not be negative")
        location = parse_s3_uri(audio_uri, expected_bucket=self._bucket)
        parent = PurePosixPath(location.key).parent
        return str(parent / "chunks" / str(chunk_index) / "audio.wav")

    def upload_chunk(self, audio_uri: str, chunk_index: int, path: Path) -> str:
        key = self.chunk_key(audio_uri, chunk_index)
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                raise OSError("chunk WAV is missing")
            self._client.upload_file(
                str(path), self._bucket, key, ExtraArgs={"ContentType": "audio/wav"}
            )
        except Exception as exc:
            raise StorageError("Unable to upload chunk audio.") from exc
        return f"s3://{self._bucket}/{key}"

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
