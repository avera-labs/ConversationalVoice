from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import boto3

from .config import EnvironmentSettings


class StorageError(RuntimeError):
    """Raised when an object storage operation fails safely."""


class InvalidS3UriError(StorageError):
    """Raised when an input URI is outside the configured storage contract."""


@dataclass(frozen=True, slots=True)
class S3Location:
    """Validated bucket and object key parsed from an absolute S3 URI."""

    bucket: str
    key: str


def parse_s3_uri(uri: str, *, expected_bucket: str) -> S3Location:
    """Parse a canonical S3 URI restricted to the configured bucket."""

    if not uri or any(character.isspace() for character in uri):
        raise InvalidS3UriError("The raw audio URI is invalid.")
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
        raise InvalidS3UriError("The raw audio URI is invalid.")

    key = parsed.path[1:]
    segments = key.split("/")
    if not key or any(segment in ("", ".", "..") for segment in segments):
        raise InvalidS3UriError("The raw audio URI is invalid.")
    return S3Location(bucket=expected_bucket, key=key)


class ObjectStorage:
    """S3 adapter for raw input and deterministic task artifacts."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def create(cls, settings: EnvironmentSettings) -> ObjectStorage:
        client_options: dict[str, str] = {"region_name": settings.s3_region}
        if settings.s3_endpoint_url is not None:
            client_options["endpoint_url"] = settings.s3_endpoint_url
        return cls(
            client=boto3.client("s3", **client_options),
            bucket=settings.s3_bucket,
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    def check_readiness(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except Exception as exc:
            raise StorageError("Object storage is not ready.") from exc

    def download_raw_audio(self, audio_uri: str, destination: Path) -> None:
        location = parse_s3_uri(audio_uri, expected_bucket=self._bucket)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(
                location.bucket,
                location.key,
                str(destination),
            )
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise StorageError("Unable to download the normalized WAV.") from exc

    @staticmethod
    def vad_segments_key(raw_audio_id: UUID) -> str:
        return f"raw_audios/{raw_audio_id}/vad_segments.json"

    @staticmethod
    def audio_part_key(raw_audio_id: UUID, part_index: int) -> str:
        if part_index < 0:
            raise ValueError("part_index must not be negative")
        return f"raw_audios/{raw_audio_id}/audio_parts/{part_index}/audio.wav"

    def upload_vad_segments(self, raw_audio_id: UUID, path: Path) -> str:
        return self._upload(
            path=path,
            key=self.vad_segments_key(raw_audio_id),
            content_type="application/json",
            safe_error="Unable to upload the VAD segments artifact.",
        )

    def upload_audio_part(
        self,
        raw_audio_id: UUID,
        part_index: int,
        path: Path,
    ) -> str:
        return self._upload(
            path=path,
            key=self.audio_part_key(raw_audio_id, part_index),
            content_type="audio/wav",
            safe_error="Unable to upload an audio part.",
        )

    def _upload(
        self,
        *,
        path: Path,
        key: str,
        content_type: str,
        safe_error: str,
    ) -> str:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                raise OSError("artifact is missing or empty")
            self._client.upload_file(
                str(path),
                self._bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        except Exception as exc:
            raise StorageError(safe_error) from exc
        return f"s3://{self._bucket}/{key}"

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
