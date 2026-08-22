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
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or parsed.netloc != self.bucket
            or not parsed.path.startswith("/")
        ):
            raise ValueError("invalid S3 URI")
        key = parsed.path[1:]
        if (
            not key
            or "%" in key
            or "\\" in key
            or any(part in {"", ".", ".."} for part in key.split("/"))
        ):
            raise ValueError("invalid S3 URI")
        return key

    def separation_uris(self, chunk_audio_uri: str) -> tuple[str, str]:
        key = PurePosixPath(self._key(chunk_audio_uri))
        base = key.parent / "results" / "separated"
        return tuple(
            f"s3://{self.bucket}/{base / f'speaker-{slot}.wav'}" for slot in range(2)
        )

    def transcription_uris(self, chunk_audio_uri: str) -> tuple[str, str]:
        key = PurePosixPath(self._key(chunk_audio_uri))
        base = key.parent / "results"
        return (
            f"s3://{self.bucket}/{base / 'transcript.json'}",
            f"s3://{self.bucket}/{base / 'word_alignment.json'}",
        )

    def reference_manifest_uri(self, audio_part_uri: str) -> str:
        key = PurePosixPath(self._key(audio_part_uri))
        return f"s3://{self.bucket}/{key.parent / 'speaker-references' / 'references.json'}"

    def reference_audio_uri(self, audio_part_uri: str, diarization_id: int) -> str:
        key = PurePosixPath(self._key(audio_part_uri))
        return f"s3://{self.bucket}/{key.parent / 'speaker-references' / f'speaker-{diarization_id}.wav'}"

    def output_uris(self, chunk_audio_uri: str) -> dict:
        key = PurePosixPath(self._key(chunk_audio_uri))
        base = key.parent / "results" / "reconstruction"
        return {
            "manifest": f"s3://{self.bucket}/{base / 'manifest.json'}",
            "transcript": f"s3://{self.bucket}/{base / 'transcript.json'}",
            "speaker_audio": tuple(
                f"s3://{self.bucket}/{base / f'speaker-{slot}.wav'}"
                for slot in range(2)
            ),
        }

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
        self._upload(uri, source, "application/json")

    def upload_wav(self, uri: str, source: Path):
        self._upload(uri, source, "audio/wav")

    def _upload(self, uri: str, source: Path, content_type: str):
        if not source.is_file() or source.stat().st_size <= 0:
            raise OSError("artifact is missing")
        self.client.upload_file(
            str(source),
            self.bucket,
            self._key(uri),
            ExtraArgs={"ContentType": content_type},
        )

    def close(self):
        close = getattr(self.client, "close", None)
        if close:
            close()
