from datetime import UTC, datetime
from uuid import UUID

from voice_pipeline_score_chunk.repository import Claim, Disposition
from voice_pipeline_score_chunk.task import Handler
from voice_pipeline_score_completed_chunks.repository import CompletedChunk


class FakeRepository:
    def __init__(self, claim):
        self.value = claim
        self.completed = None

    def claim(self, identifier, *, model_fingerprint):
        assert model_fingerprint == "model-fingerprint"
        return self.value

    def complete(self, identifier, **kwargs):
        self.completed = (identifier, kwargs)
        return kwargs["evaluation"]


class FakeStorage:
    def upload_artifacts(self, chunk_audio_uri, **kwargs):
        assert chunk_audio_uri == "s3://bucket/chunk.wav"
        return {
            "score-report.json": {
                "uri": "s3://bucket/report.json",
                "size_bytes": 10,
                "sha256": "a" * 64,
            }
        }


class FakeService:
    model_fingerprint = "model-fingerprint"

    def score(self, chunk):
        return (
            {
                "status": "success",
                "generated_at": "2026-01-01T00:00:00+00:00",
            },
            {"score-report.json": b"report"},
        )


def test_handler_scores_and_persists_only_evaluation_descriptor() -> None:
    identifier = UUID(int=1)
    chunk = CompletedChunk(
        identifier,
        "en",
        datetime.now(UTC),
        "s3://bucket/part.wav",
        {},
        chunk_audio_uri="s3://bucket/chunk.wav",
    )
    repository = FakeRepository(Claim(Disposition.READY, chunk, "source-fingerprint"))
    result = Handler(repository, FakeStorage(), FakeService())(str(identifier))
    assert result["outcome"] == "completed"
    evaluation = repository.completed[1]["evaluation"]
    assert evaluation["gpu_used"] is False
    assert evaluation["local_asr_weights"] is False
    assert set(evaluation) == {
        "schema_version",
        "status",
        "language",
        "model_fingerprint",
        "source_fingerprint",
        "generated_at",
        "gpu_used",
        "local_asr_weights",
        "report",
        "artifacts",
    }


def test_handler_skips_matching_existing_evaluation() -> None:
    evaluation = {"status": "complete"}
    repository = FakeRepository(
        Claim(Disposition.ALREADY_SCORED, None, "source", evaluation)
    )
    result = Handler(repository, FakeStorage(), FakeService())(str(UUID(int=2)))
    assert result["outcome"] == "already_scored"
    assert result["evaluation"] == evaluation
