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
        if not isinstance(uri, str) or not uri or any(char.isspace() for char in uri):
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

    def transcription_uri(self, chunk_audio_uri: str) -> str:
        key = PurePosixPath(self._key(chunk_audio_uri))
        return f"s3://{self.bucket}/{key.parent / 'results' / 'transcript.json'}"

    def separation_uris(self, chunk_audio_uri: str) -> tuple[str, str]:
        key = PurePosixPath(self._key(chunk_audio_uri))
        base = key.parent / "results" / "separated"
        return tuple(
            f"s3://{self.bucket}/{base / f'speaker-{speaker_id}.wav'}"
            for speaker_id in range(2)
        )

    def transcription_uris(self, chunk_audio_uri: str) -> tuple[str, str]:
        key = PurePosixPath(self._key(chunk_audio_uri))
        base = key.parent / "results"
        return (
            f"s3://{self.bucket}/{base / 'transcript.json'}",
            f"s3://{self.bucket}/{base / 'word_alignment.json'}",
        )

    def persona_uri(self, chunk_audio_uri: str) -> str:
        key = PurePosixPath(self._key(chunk_audio_uri))
        return f"s3://{self.bucket}/{key.parent / 'results' / 'persona.json'}"

    def reference_manifest_uri(self, audio_part_uri: str) -> str:
        key = PurePosixPath(self._key(audio_part_uri))
        path = key.parent / "speaker-references" / "references.json"
        return f"s3://{self.bucket}/{path}"

    def reference_audio_uri(
        self, audio_part_uri: str, diarization_speaker_id: int
    ) -> str:
        if (
            isinstance(diarization_speaker_id, bool)
            or not isinstance(diarization_speaker_id, int)
            or diarization_speaker_id < 0
        ):
            raise ValueError("invalid diarization speaker ID")
        key = PurePosixPath(self._key(audio_part_uri))
        path = (
            key.parent / "speaker-references" / f"speaker-{diarization_speaker_id}.wav"
        )
        return f"s3://{self.bucket}/{path}"

    def output_uris(self, chunk_audio_uri: str) -> dict[str, object]:
        key = PurePosixPath(self._key(chunk_audio_uri))
        base = key.parent / "results" / "dialogue-extension"
        return {
            "script": f"s3://{self.bucket}/{base / 'script.json'}",
            "transcript": f"s3://{self.bucket}/{base / 'transcript.json'}",
            "speaker_audio": tuple(
                f"s3://{self.bucket}/{base / f'speaker-{speaker_id}.wav'}"
                for speaker_id in range(2)
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

    def upload_json(self, uri: str, source: Path) -> None:
        self._upload(uri, source, "application/json")

    def upload_wav(self, uri: str, source: Path) -> None:
        self._upload(uri, source, "audio/wav")

    def _upload(self, uri: str, source: Path, content_type: str) -> None:
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
