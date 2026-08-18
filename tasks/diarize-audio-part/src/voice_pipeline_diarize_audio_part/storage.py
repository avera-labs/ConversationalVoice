"""Canonical S3 input and deterministic artifact storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import boto3

from .config import EnvironmentSettings


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
        raise InvalidS3UriError("Input artifact URI is invalid.")
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
        raise InvalidS3UriError("Input artifact URI is invalid.")
    key = parsed.path[1:]
    if not key or any(segment in {"", ".", ".."} for segment in key.split("/")):
        raise InvalidS3UriError("Input artifact URI is invalid.")
    return S3Location(parsed.netloc, key)


class ObjectStorage:
    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def create(cls, settings: EnvironmentSettings) -> ObjectStorage:
        options: dict[str, str] = {"region_name": settings.s3_region}
        if settings.s3_endpoint_url is not None:
            options["endpoint_url"] = settings.s3_endpoint_url
        return cls(boto3.client("s3", **options), settings.s3_bucket)

    def check_readiness(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except Exception as exc:
            raise StorageError("Object storage is not ready.") from exc

    def download_audio(self, audio_uri: str, destination: Path) -> int:
        location = parse_s3_uri(audio_uri, expected_bucket=self._bucket)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(location.bucket, location.key, str(destination))
            size = destination.stat().st_size
            if size <= 0:
                raise OSError("empty object")
            return size
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise StorageError("Unable to download input audio.") from exc

    def artifact_key(self, audio_uri: str) -> str:
        location = parse_s3_uri(audio_uri, expected_bucket=self._bucket)
        return str(PurePosixPath(location.key).with_name("diarization.json"))

    def reference_manifest_key(self, audio_uri: str) -> str:
        location = parse_s3_uri(audio_uri, expected_bucket=self._bucket)
        return str(
            PurePosixPath(location.key).parent
            / "speaker-references"
            / "references.json"
        )

    def reference_audio_key(self, audio_uri: str, speaker_id: int) -> str:
        if (
            isinstance(speaker_id, bool)
            or not isinstance(speaker_id, int)
            or speaker_id < 0
        ):
            raise ValueError("speaker ID must be a non-negative integer")
        location = parse_s3_uri(audio_uri, expected_bucket=self._bucket)
        return str(
            PurePosixPath(location.key).parent
            / "speaker-references"
            / f"speaker-{speaker_id}.wav"
        )

    def reference_audio_uri(self, audio_uri: str, speaker_id: int) -> str:
        return f"s3://{self._bucket}/{self.reference_audio_key(audio_uri, speaker_id)}"

    def _upload(self, path: Path, key: str, *, content_type: str) -> str:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                raise OSError("artifact missing")
            self._client.upload_file(
                str(path),
                self._bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        except Exception as exc:
            raise StorageError("Unable to upload task artifact.") from exc
        return f"s3://{self._bucket}/{key}"

    def upload_artifact(self, audio_uri: str, path: Path) -> str:
        key = self.artifact_key(audio_uri)
        return self._upload(path, key, content_type="application/json")

    def upload_reference_audio(
        self, audio_uri: str, speaker_id: int, path: Path
    ) -> str:
        key = self.reference_audio_key(audio_uri, speaker_id)
        return self._upload(path, key, content_type="audio/wav")

    def upload_reference_manifest(self, audio_uri: str, path: Path) -> str:
        key = self.reference_manifest_key(audio_uri)
        return self._upload(path, key, content_type="application/json")

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
