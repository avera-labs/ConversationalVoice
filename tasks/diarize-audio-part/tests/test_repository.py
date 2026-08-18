from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql

from voice_pipeline_diarize_audio_part.repository import (
    AudioPartNotFoundError,
    ClaimDisposition,
    DiarizationRepository,
    InvalidAudioPartStatusError,
    _claim_statement,
    normalize_database_url,
)


def test_claim_is_one_conditional_update_and_returns_only_required_input() -> None:
    statement = str(
        _claim_statement(UUID("11111111-1111-1111-1111-111111111111")).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "status IN ('pending', 'failed')" in statement
    assert (
        "status='diarizing'" in statement.replace(" ", "")
        or "status = 'diarizing'" in statement
    )
    assert "diarization_uri=NULL" in statement.replace(" ", "")
    assert (
        "RETURNING audio_parts.id, audio_parts.audio_uri, audio_parts.duration_ms"
        in statement
    )


def test_generic_postgres_url_uses_psycopg_three() -> None:
    assert (
        normalize_database_url("postgresql://host/db") == "postgresql+psycopg://host/db"
    )
    assert (
        normalize_database_url("postgres://host/db") == "postgresql+psycopg://host/db"
    )


class Result:
    def __init__(self, row):
        self.row = row

    def one_or_none(self):
        return self.row


class Session:
    def __init__(self, claimed, status):
        self.claimed = claimed
        self.status = status

    def execute(self, _statement):
        return Result(self.claimed)

    def scalar(self, _statement):
        return self.status


class Transaction:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, *_args):
        return None


class Factory:
    def __init__(self, session):
        self.session = session

    def begin(self):
        return Transaction(self.session)


def repository(claimed=None, status=None):
    return DiarizationRepository(Factory(Session(claimed, status)))


def test_successful_claim_returns_only_persisted_input_contract() -> None:
    claimed = SimpleNamespace(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        audio_uri="s3://bucket/audio.wav",
        duration_ms=1000,
    )
    result = repository(claimed=claimed).claim(claimed.id)
    assert result.disposition is ClaimDisposition.CLAIMED
    assert result.audio_uri == "s3://bucket/audio.wav"
    assert result.duration_ms == 1000


@pytest.mark.parametrize(
    ("status", "disposition"),
    [
        ("diarizing", ClaimDisposition.ALREADY_PROCESSING),
        ("filtering", ClaimDisposition.ALREADY_PROCESSING),
        ("diarized", ClaimDisposition.DISPATCH_READY),
        ("completed", ClaimDisposition.COMPLETED),
    ],
)
def test_unclaimed_status_only_branches(
    status: str, disposition: ClaimDisposition
) -> None:
    result = repository(status=status).claim(
        UUID("11111111-1111-1111-1111-111111111111")
    )
    assert result.disposition is disposition
    assert result.audio_uri is None
    assert result.duration_ms is None


def test_missing_and_invalid_status_are_rejected() -> None:
    identifier = UUID("11111111-1111-1111-1111-111111111111")
    with pytest.raises(AudioPartNotFoundError):
        repository(status=None).claim(identifier)
    with pytest.raises(InvalidAudioPartStatusError):
        repository(status="unexpected").claim(identifier)
