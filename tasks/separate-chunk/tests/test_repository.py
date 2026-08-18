from types import SimpleNamespace
from uuid import UUID

import pytest
from voice_pipeline_separate_chunk.repository import Disposition, Repository

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")


class Session:
    def __init__(self, row):
        self.row = row

    def scalar(self, _statement):
        return self.row


class Transaction:
    def __init__(self, row):
        self.session = Session(row)

    def __enter__(self):
        return self.session

    def __exit__(self, *_args):
        return None


class Factory:
    def __init__(self, row):
        self.row = row

    def begin(self):
        return Transaction(self.row)


def row(status, separation):
    return SimpleNamespace(
        id=IDENTIFIER,
        status=status,
        audio_part_id=UUID("22222222-2222-2222-2222-222222222222"),
        audio_uri="s3://test/chunks/1/chunk.wav",
        lang="en",
        duration_ms=20_000,
        relative_start_ms=0,
        relative_end_ms=20_000,
        diarizations={"snapshot": "present"},
        final_results={"separation": separation} if separation is not None else {},
        error=None,
    )


@pytest.mark.parametrize(
    "status",
    [
        "persona_generating",
        "persona_generated",
        "extending",
        "completed",
        "rejected",
    ],
)
def test_claim_treats_downstream_persona_states_as_already_separated(status):
    claim = Repository(Factory(row(status, {"result": "present"}))).claim(IDENTIFIER)

    assert claim.disposition is Disposition.ALREADY_SEPARATED
    assert claim.status == status
    assert claim.separation == {"result": "present"}


@pytest.mark.parametrize(
    "status", ["persona_generating", "persona_generated", "extending", "completed"]
)
def test_claim_rejects_downstream_persona_states_without_separation(status):
    with pytest.raises(RuntimeError, match="invalid_chunk_state"):
        Repository(Factory(row(status, None))).claim(IDENTIFIER)


@pytest.mark.parametrize(
    "status",
    [
        "persona_generating",
        "persona_generated",
        "extending",
        "completed",
        "rejected",
    ],
)
def test_claim_rejects_non_object_separation_in_downstream_persona_states(status):
    with pytest.raises(RuntimeError, match="invalid_separation_state"):
        Repository(Factory(row(status, "invalid"))).claim(IDENTIFIER)
