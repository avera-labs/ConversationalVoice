from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import urlsplit

import boto3

from .errors import ScoringError


@dataclass(frozen=True, slots=True)
class StoredObject:
    uri: str
    size_bytes: int
    sha256: str


def parse_identity(value: object, *, name: str) -> StoredObject:
    if not isinstance(value, dict):
        raise ScoringError("invalid_artifact_identity", f"{name} is not an object")
    uri = value.get("uri")
    size = value.get("size_bytes")
    sha256 = value.get("sha256")
    if (
        not isinstance(uri, str)
        or not uri
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise ScoringError("invalid_artifact_identity", f"{name} is invalid")
    return StoredObject(uri, size, sha256)


class ObjectStorage:
    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        endpoint_url: str | None = None,
    ):
        options: dict[str, str] = {"region_name": region}
        if endpoint_url:
            options["endpoint_url"] = endpoint_url
        self.bucket = bucket
        self.client = boto3.client("s3", **options)

    def _key(self, uri: str) -> str:
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or parsed.netloc != self.bucket
            or not parsed.path.startswith("/")
            or parsed.query
            or parsed.fragment
            or "%" in parsed.path
            or "\\" in parsed.path
        ):
            raise ScoringError("invalid_s3_uri")
        key = parsed.path[1:]
        if not key or any(part in {"", ".", ".."} for part in key.split("/")):
            raise ScoringError("invalid_s3_uri")
        return key

    def download(self, identity: StoredObject) -> bytes:
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=self._key(identity.uri)
            )
            body = response["Body"]
            try:
                payload = body.read()
            finally:
                body.close()
        except ScoringError:
            raise
        except Exception as exc:
            raise ScoringError("artifact_download_failed") from exc
        if len(payload) != identity.size_bytes:
            raise ScoringError("artifact_size_mismatch")
        if hashlib.sha256(payload).hexdigest() != identity.sha256:
            raise ScoringError("artifact_sha256_mismatch")
        return payload

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()
