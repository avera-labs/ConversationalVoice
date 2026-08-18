from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql

from voice_pipeline_quality_filter_audio_part.repository import (
    AudioPartClaim,
    AudioPartNotFoundError,
    ClaimDisposition,
    InvalidAudioPartStatusError,
    PersistedChunk,
    PersistenceConflictError,
    QualityFilterRepository,
    _validate_planned_chunks,
    _claim_statement,
    normalize_database_url,
)


IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")


def test_claim_is_one_compare_and_set_update() -> None:
    statement = str(
        _claim_statement(IDENTIFIER).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "status IN ('diarized', 'failed')" in statement
    assert "status='filtering'" in statement.replace(" ", "")
    assert "error=NULL" in statement.replace(" ", "")
    assert "audio_parts.diarization_uri" in statement


def test_postgres_url_uses_psycopg_three() -> None:
    assert normalize_database_url("postgresql://host/db") == "postgresql+psycopg://host/db"


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


def repository(*, claimed=None, status=None) -> QualityFilterRepository:
    return QualityFilterRepository(Factory(Session(claimed, status)))


def test_claim_returns_all_stable_inputs() -> None:
    row = SimpleNamespace(
        id=IDENTIFIER,
        audio_uri="s3://bucket/audio.wav",
        diarization_uri="s3://bucket/diarization.json",
        duration_ms=1000,
        lang="en",
    )
    claim = repository(claimed=row).claim(IDENTIFIER)
    assert claim.disposition is ClaimDisposition.CLAIMED
    assert claim.diarization_uri == row.diarization_uri


@pytest.mark.parametrize(
    ("status", "disposition"),
    [
        ("filtering", ClaimDisposition.ALREADY_PROCESSING),
        ("diarizing", ClaimDisposition.ALREADY_PROCESSING),
        ("completed", ClaimDisposition.ALREADY_COMPLETED),
        ("pending", ClaimDisposition.NOT_READY),
    ],
)
def test_no_op_status_mapping(status: str, disposition: ClaimDisposition) -> None:
    assert repository(status=status).claim(IDENTIFIER).disposition is disposition


def test_missing_and_invalid_status_are_rejected() -> None:
    with pytest.raises(AudioPartNotFoundError):
        repository(status=None).claim(IDENTIFIER)
    with pytest.raises(InvalidAudioPartStatusError):
        repository(status="unexpected").claim(IDENTIFIER)


CLAIM = AudioPartClaim(
    IDENTIFIER,
    ClaimDisposition.CLAIMED,
    "filtering",
    "s3://bucket/parts/0/audio.wav",
    "s3://bucket/parts/0/diarization.json",
    30000,
    "en",
)
CHUNK_ID = UUID("22222222-2222-2222-2222-222222222222")


def planned_chunk(**overrides) -> PersistedChunk:
    values = {
        "chunk_id": CHUNK_ID,
        "chunk_index": 0,
        "audio_uri": "s3://bucket/parts/0/chunks/0/audio.wav",
        "lang": "en",
        "duration_ms": 20000,
        "start_ms": 0,
        "end_ms": 20000,
    }
    values.update(overrides)
    return PersistedChunk(**values)


@pytest.mark.parametrize(
    "chunk",
    [
        planned_chunk(chunk_index=1),
        planned_chunk(audio_uri=""),
        planned_chunk(lang="zh"),
        planned_chunk(duration_ms=0),
        planned_chunk(duration_ms=19999),
        planned_chunk(start_ms=-1, duration_ms=20001),
        planned_chunk(end_ms=30001, duration_ms=30001),
    ],
)
def test_invalid_planned_chunks_are_rejected_before_transaction(chunk) -> None:
    with pytest.raises(PersistenceConflictError):
        _validate_planned_chunks(CLAIM, (chunk,))


class ScalarRows:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class CompletionSession:
    def __init__(self, part, existing=()) -> None:
        self.part = part
        self.existing = existing
        self.added = []

    def scalar(self, _statement):
        return self.part

    def scalars(self, _statement):
        return ScalarRows(self.existing)

    def add(self, value):
        self.added.append(value)


def completion_repository(part, existing=()):
    session = CompletionSession(part, existing)
    return QualityFilterRepository(Factory(session)), session


def claimed_part(**overrides):
    values = {
        "status": "filtering",
        "audio_uri": CLAIM.audio_uri,
        "diarization_uri": CLAIM.diarization_uri,
        "duration_ms": CLAIM.duration_ms,
        "lang": CLAIM.lang,
        "error": "stale",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def existing_chunk(**overrides):
    values = {
        "id": CHUNK_ID,
        "chunk_index": 0,
        "status": "pending",
        "audio_uri": "s3://bucket/parts/0/chunks/0/audio.wav",
        "lang": "en",
        "duration_ms": 20000,
        "relative_start_ms": 0,
        "relative_end_ms": 20000,
        "error": None,
        "diarization_model": None,
        "diarizations": None,
        "persona": None,
        "final_results": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_complete_inserts_all_chunks_and_completes_part_atomically() -> None:
    part = claimed_part()
    repo, session = completion_repository(part)
    result = repo.complete(CLAIM, (planned_chunk(),))
    assert result == (CHUNK_ID,)
    assert len(session.added) == 1
    assert session.added[0].audio_part_id == IDENTIFIER
    assert part.status == "completed"
    assert part.error is None


def test_complete_accepts_identical_existing_chunks_without_replacing_ids() -> None:
    part = claimed_part()
    repo, session = completion_repository(part, (existing_chunk(),))
    assert repo.complete(CLAIM, (planned_chunk(),)) == (CHUNK_ID,)
    assert session.added == []
    assert part.status == "completed"


@pytest.mark.parametrize(
    "existing",
    [
        (existing_chunk(audio_uri="s3://bucket/conflict.wav"),),
        (existing_chunk(final_results={"consumed": True}),),
        (existing_chunk(), existing_chunk(id=UUID("33333333-3333-3333-3333-333333333333"))),
    ],
)
def test_complete_rejects_existing_chunk_conflicts(existing) -> None:
    repo, _session = completion_repository(claimed_part(), existing)
    with pytest.raises(PersistenceConflictError):
        repo.complete(CLAIM, (planned_chunk(),))


def test_complete_zero_chunks_is_success() -> None:
    part = claimed_part()
    repo, session = completion_repository(part)
    assert repo.complete(CLAIM, ()) == ()
    assert session.added == []
    assert part.status == "completed"


def test_complete_rechecks_claimed_part_identity() -> None:
    repo, _session = completion_repository(claimed_part(lang="zh"))
    with pytest.raises(PersistenceConflictError):
        repo.complete(CLAIM, (planned_chunk(),))
