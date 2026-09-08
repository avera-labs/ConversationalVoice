from datetime import UTC, datetime
from uuid import UUID

from voice_pipeline_score_chunk.repository import (
    Disposition,
    Repository,
    source_fingerprint,
)


def test_source_fingerprint_excludes_previous_evaluation() -> None:
    left = {"separation": {"value": 1}, "evaluation": {"old": True}}
    right = {"evaluation": {"new": True}, "separation": {"value": 1}}
    assert source_fingerprint(left) == source_fingerprint(right)


class Result:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class Connection:
    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, *_args):
        return Result(self.row)


def completed_row(results):
    return {
        "id": UUID(int=1),
        "status": "completed",
        "lang": "en",
        "updated_at": datetime.now(UTC),
        "audio_uri": "s3://bucket/chunk.wav",
        "audio_part_audio_uri": "s3://bucket/part.wav",
        "source_cluster_id": UUID(int=2),
        "final_results": results,
    }


def test_claim_skips_matching_evaluation(monkeypatch) -> None:
    results = {"separation": {"value": 1}}
    fingerprint = source_fingerprint(results)
    results["evaluation"] = {
        "model_fingerprint": "model",
        "source_fingerprint": fingerprint,
    }
    monkeypatch.setattr(
        "voice_pipeline_score_chunk.repository.psycopg.connect",
        lambda *_args, **_kwargs: Connection(completed_row(results)),
    )
    claim = Repository("postgresql://database").claim(
        UUID(int=1), model_fingerprint="model"
    )
    assert claim.disposition is Disposition.ALREADY_SCORED
    assert claim.chunk is None


def test_claim_returns_completed_chunk_for_new_model(monkeypatch) -> None:
    row = completed_row({"separation": {"value": 1}})
    monkeypatch.setattr(
        "voice_pipeline_score_chunk.repository.psycopg.connect",
        lambda *_args, **_kwargs: Connection(row),
    )
    claim = Repository("postgresql://database").claim(
        UUID(int=1), model_fingerprint="new-model"
    )
    assert claim.disposition is Disposition.READY
    assert claim.chunk is not None
    assert claim.chunk.chunk_audio_uri == "s3://bucket/chunk.wav"
