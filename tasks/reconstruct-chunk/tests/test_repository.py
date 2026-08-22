from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy.sql.dml import Update

from voice_pipeline_reconstruct_chunk.repository import Disposition, Repository

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")
PART = UUID("22222222-2222-2222-2222-222222222222")


def row(status, reconstruction=None):
    results = {
        "separation": {"valid": "separation"},
        "transcription": {"valid": "transcription"},
        "persona": {"valid": "persona"},
    }
    if reconstruction is not None:
        results["reconstruction"] = reconstruction
    return SimpleNamespace(
        id=IDENTIFIER,
        audio_part_id=PART,
        status=status,
        error="old",
        audio_uri="s3://bucket/part/chunks/0/audio.wav",
        lang="en",
        duration_ms=1000,
        diarizations={"valid": "snapshot"},
        persona={"valid": "persona"},
        final_results=results,
    )


class Result:
    def __init__(self, value):
        self.value = value

    def one_or_none(self):
        return self.value


class Session:
    def __init__(self, record):
        self.record = record

    def execute(self, statement):
        if isinstance(statement, Update):
            return SimpleNamespace(rowcount=1)
        return Result((self.record, "s3://bucket/part/audio.wav"))

    def get(self, _model, _identifier, **_kwargs):
        return self.record


class Transaction:
    def __init__(self, record):
        self.session = Session(record)

    def __enter__(self):
        return self.session

    def __exit__(self, *_args):
        return None


class Factory:
    def __init__(self, record):
        self.record = record

    def begin(self):
        return Transaction(self.record)


@pytest.mark.parametrize("status", ["persona_generated", "failed"])
def test_claim_advances_reconstruction(status):
    record = row(status)
    claim = Repository(Factory(record)).claim(IDENTIFIER)
    assert claim.disposition is Disposition.CLAIMED
    assert record.status == "reconstructing"


def test_completed_reconstruction_is_ready_to_dispatch():
    reconstruction = {"valid": "reconstruction"}
    claim = Repository(Factory(row("reconstructed", reconstruction))).claim(IDENTIFIER)
    assert claim.disposition is Disposition.READY_TO_DISPATCH
    assert claim.reconstruction == reconstruction


def test_completion_preserves_upstream_namespaces():
    record = row("persona_generated")
    repository = Repository(Factory(record))
    claim = repository.claim(IDENTIFIER)
    result = {"valid": "reconstruction"}
    repository.complete(claim, result)
    assert record.status == "reconstructed"
    assert record.final_results["reconstruction"] == result
    assert record.final_results["persona"] == {"valid": "persona"}
