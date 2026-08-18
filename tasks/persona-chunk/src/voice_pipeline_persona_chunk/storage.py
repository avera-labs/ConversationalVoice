from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import boto3


class ObjectStorage:
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    @classmethod
    def create(cls, environment):
        options = {"region_name": environment.s3_region}
        if environment.s3_endpoint_url:
            options["endpoint_url"] = environment.s3_endpoint_url
        return cls(boto3.client("s3", **options), environment.s3_bucket)

    def _key(self, uri: str) -> str:
        if not isinstance(uri, str) or not uri or any(c.isspace() for c in uri):
            raise ValueError("invalid S3 URI")
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or parsed.netloc != self.bucket
            or not parsed.path.startswith("/")
            or parsed.path.startswith("//")
            or parsed.query
            or parsed.fragment
            or "@" in parsed.netloc
            or ":" in parsed.netloc
            or "%" in parsed.path
            or "\\" in parsed.path
            or any(part in {"", ".", ".."} for part in parsed.path[1:].split("/"))
        ):
            raise ValueError("invalid S3 URI")
        return parsed.path[1:]

    def speaker_uris(self, audio_uri: str) -> tuple[str, str]:
        key = PurePosixPath(self._key(audio_uri))
        base = key.parent / "results" / "separated"
        return tuple(
            f"s3://{self.bucket}/{base}/speaker-{slot}.wav" for slot in range(2)
        )

    def transcription_uris(self, audio_uri: str) -> tuple[str, str]:
        key = PurePosixPath(self._key(audio_uri))
        base = key.parent / "results"
        return (
            f"s3://{self.bucket}/{base}/transcript.json",
            f"s3://{self.bucket}/{base}/word_alignment.json",
        )

    def persona_uri(self, audio_uri: str) -> str:
        key = PurePosixPath(self._key(audio_uri))
        return f"s3://{self.bucket}/{key.parent / 'results' / 'persona.json'}"

    def download(self, uri: str, destination: Path) -> int:
        try:
            self.client.download_file(self.bucket, self._key(uri), str(destination))
            size = destination.stat().st_size
            if size <= 0:
                raise OSError("downloaded object is empty")
            return size
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def upload_json(self, uri: str, source: Path):
        self.client.upload_file(
            str(source),
            self.bucket,
            self._key(uri),
            ExtraArgs={"ContentType": "application/json"},
        )

    def close(self):
        close = getattr(self.client, "close", None)
        if close:
            close()
