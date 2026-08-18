from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy.sql.dml import Update

from voice_pipeline_extend_chunk.repository import Disposition, Repository

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")
PART = UUID("22222222-2222-2222-2222-222222222222")


def row(status, extension=None):
    results = {
        "separation": {"separation": "valid"},
        "transcription": {"transcription": "valid"},
        "persona": {"persona": "valid"},
    }
    if extension is not None:
        results["dialogue_extension"] = extension
    return SimpleNamespace(
        id=IDENTIFIER,
        audio_part_id=PART,
        status=status,
        error="old error",
        audio_uri="s3://bucket/part/chunks/0/audio.wav",
        lang="en",
        duration_ms=1000,
        diarizations={"snapshot": "valid"},
        persona={"document": "valid"},
        final_results=results,
    )


class QueryResult:
    def __init__(self, value):
        self.value = value

    def one_or_none(self):
        return self.value


class UpdateResult:
    rowcount = 1


class Session:
    def __init__(self, record):
        self.record = record

    def execute(self, statement):
        if isinstance(statement, Update):
            return UpdateResult()
        return QueryResult((self.record, "s3://bucket/part/audio.wav"))


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
def test_claim_advances_eligible_extension_state(status):
    record = row(status)
    claim = Repository(Factory(record)).claim(IDENTIFIER)

    assert claim.disposition is Disposition.CLAIMED
    assert record.status == "extending"
    assert record.error is None
    assert claim.audio_part_audio_uri == "s3://bucket/part/audio.wav"


def test_completed_requires_and_returns_extension_namespace():
    extension = {"result": "valid"}
    claim = Repository(Factory(row("completed", extension))).claim(IDENTIFIER)

    assert claim.disposition is Disposition.ALREADY_COMPLETED
    assert claim.extension_result == extension


def test_completed_without_extension_namespace_is_invalid():
    with pytest.raises(RuntimeError, match="invalid_completed_extension_state"):
        Repository(Factory(row("completed"))).claim(IDENTIFIER)


@pytest.mark.parametrize(
    ("status", "disposition"),
    [
        ("extending", Disposition.ALREADY_PROCESSING),
        ("rejected", Disposition.ALREADY_REJECTED),
    ],
)
def test_terminal_or_in_progress_invocation_is_a_no_op(status, disposition):
    claim = Repository(Factory(row(status))).claim(IDENTIFIER)
    assert claim.disposition is disposition


def test_completion_preserves_other_result_namespaces():
    record = row("persona_generated")
    repository = Repository(Factory(record))
    claim = repository.claim(IDENTIFIER)
    result = {"result": "extension"}

    repository.complete(claim, result)

    assert record.status == "completed"
    assert record.error is None
    assert record.final_results["dialogue_extension"] == result
    assert record.final_results["persona"] == {"persona": "valid"}
