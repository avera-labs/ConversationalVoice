from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import boto3
from voice_pipeline_score_completed_chunks.storage import StoredObject


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
            or parsed.query
            or parsed.fragment
            or "%" in parsed.path
            or "\\" in parsed.path
        ):
            raise ValueError("invalid S3 URI")
        key = parsed.path[1:]
        if not key or any(part in {"", ".", ".."} for part in key.split("/")):
            raise ValueError("invalid S3 URI")
        return key

    def download(self, identity: StoredObject) -> bytes:
        response = self.client.get_object(
            Bucket=self.bucket, Key=self._key(identity.uri)
        )
        body = response["Body"]
        try:
            payload = body.read()
        finally:
            body.close()
        if len(payload) != identity.size_bytes:
            raise RuntimeError("artifact_size_mismatch")
        if hashlib.sha256(payload).hexdigest() != identity.sha256:
            raise RuntimeError("artifact_sha256_mismatch")
        return payload

    def artifact_uri(
        self,
        chunk_audio_uri: str,
        *,
        model_fingerprint: str,
        source_fingerprint: str,
        filename: str,
    ) -> str:
        chunk_key = PurePosixPath(self._key(chunk_audio_uri))
        path = (
            chunk_key.parent
            / "results"
            / "evaluation"
            / "v2"
            / model_fingerprint
            / source_fingerprint
            / filename
        )
        return f"s3://{self.bucket}/{path}"

    def upload_artifacts(
        self,
        chunk_audio_uri: str,
        *,
        model_fingerprint: str,
        source_fingerprint: str,
        artifacts: dict[str, bytes],
    ) -> dict[str, dict[str, object]]:
        output: dict[str, dict[str, object]] = {}
        for filename, payload in sorted(artifacts.items()):
            if not payload:
                continue
            uri = self.artifact_uri(
                chunk_audio_uri,
                model_fingerprint=model_fingerprint,
                source_fingerprint=source_fingerprint,
                filename=filename,
            )
            digest = hashlib.sha256(payload).hexdigest()
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._key(uri),
                Body=payload,
                ContentType=(
                    "application/json"
                    if filename.endswith(".json")
                    else "application/x-ndjson"
                    if filename.endswith(".jsonl")
                    else "text/csv"
                ),
                Metadata={"sha256": digest},
            )
            output[filename] = {
                "uri": uri,
                "size_bytes": len(payload),
                "sha256": digest,
            }
        return output

    def close(self):
        close = getattr(self.client, "close", None)
        if close:
            close()
