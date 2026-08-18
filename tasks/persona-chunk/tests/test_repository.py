from types import SimpleNamespace
from uuid import UUID

import pytest
from voice_pipeline_persona_chunk.repository import Disposition, Repository

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")


def row(status, *, durable_persona):
    results = {
        "separation": {"separation": "valid"},
        "transcription": {"transcription": "valid"},
    }
    persona = {"persona": "valid"} if durable_persona else None
    if durable_persona:
        results["persona"] = {"result": "valid"}
    return SimpleNamespace(
        id=IDENTIFIER,
        status=status,
        audio_part_id=UUID("22222222-2222-2222-2222-222222222222"),
        audio_uri="s3://bucket/part/chunks/0/audio.wav",
        lang="en",
        duration_ms=1000,
        relative_start_ms=0,
        relative_end_ms=1000,
        diarizations={"snapshot": "valid"},
        persona=persona,
        final_results=results,
        error=None,
    )


class Session:
    def __init__(self, record):
        self.record = record

    def scalar(self, _statement):
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
def test_durable_persona_is_ready_to_dispatch(status):
    claim = Repository(Factory(row(status, durable_persona=True))).claim(IDENTIFIER)
    assert claim.disposition is Disposition.READY_TO_DISPATCH


@pytest.mark.parametrize("status", ["extending", "completed", "rejected"])
def test_downstream_state_with_persona_is_already_completed(status):
    claim = Repository(Factory(row(status, durable_persona=True))).claim(IDENTIFIER)
    assert claim.disposition is Disposition.ALREADY_COMPLETED


def test_separation_rejection_without_persona_remains_already_rejected():
    claim = Repository(Factory(row("rejected", durable_persona=False))).claim(
        IDENTIFIER
    )
    assert claim.disposition is Disposition.ALREADY_REJECTED
