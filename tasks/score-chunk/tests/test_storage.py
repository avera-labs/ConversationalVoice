import hashlib
import io

from voice_pipeline_score_chunk.storage import ObjectStorage
from voice_pipeline_score_completed_chunks.storage import StoredObject


class Body(io.BytesIO):
    def close(self):
        super().close()


class Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, *, Bucket, Key):
        return {"Body": Body(self.objects[(Bucket, Key)])}


def test_storage_uploads_deterministic_positive_size_artifacts() -> None:
    client = Client()
    storage = ObjectStorage(client, "bucket")
    artifacts = storage.upload_artifacts(
        "s3://bucket/chunks/one/audio.wav",
        model_fingerprint="a" * 64,
        source_fingerprint="b" * 64,
        artifacts={"score-report.json": b"{}\n", "failures.jsonl": b"\n"},
    )
    assert set(artifacts) == {"score-report.json", "failures.jsonl"}
    report = artifacts["score-report.json"]
    assert report["uri"].endswith(
        f"/evaluation/v2/{'a' * 64}/{'b' * 64}/score-report.json"
    )
    assert report["size_bytes"] == 3
    assert report["sha256"] == hashlib.sha256(b"{}\n").hexdigest()


def test_storage_download_verifies_identity() -> None:
    client = Client()
    client.objects[("bucket", "input.wav")] = b"audio"
    storage = ObjectStorage(client, "bucket")
    identity = StoredObject(
        "s3://bucket/input.wav", 5, hashlib.sha256(b"audio").hexdigest()
    )
    assert storage.download(identity) == b"audio"
