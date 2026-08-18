from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import boto3

from ..config import EnvironmentSettings


class ObjectStorage:
    """S3 client wrapper owned by the ingest service."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def create(cls, settings: EnvironmentSettings) -> ObjectStorage:
        client_options: dict[str, str] = {
            "region_name": settings.s3_region,
        }
        if settings.s3_endpoint_url is not None:
            client_options["endpoint_url"] = settings.s3_endpoint_url

        client = boto3.client("s3", **client_options)
        return cls(client=client, bucket=settings.s3_bucket)

    @property
    def bucket(self) -> str:
        return self._bucket

    def check_readiness(self) -> None:
        self._client.head_bucket(Bucket=self._bucket)

    @staticmethod
    def normalized_audio_key(raw_audio_id: UUID) -> str:
        return f"raw_audios/{raw_audio_id}/audio.wav"

    def upload_normalized_audio(self, raw_audio_id: UUID, path: Path) -> str:
        key = self.normalized_audio_key(raw_audio_id)
        self._client.upload_file(
            str(path),
            self._bucket,
            key,
            ExtraArgs={"ContentType": "audio/wav"},
        )
        return f"s3://{self._bucket}/{key}"

    def delete_normalized_audio(self, raw_audio_id: UUID) -> None:
        self._client.delete_object(
            Bucket=self._bucket,
            Key=self.normalized_audio_key(raw_audio_id),
        )
