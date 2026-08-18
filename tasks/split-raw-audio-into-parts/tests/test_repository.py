from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from voice_pipeline_split_raw_audio_into_parts import repository as repository_module
from voice_pipeline_split_raw_audio_into_parts.repository import (
    AudioPartDraft,
    ClaimDisposition,
    InvalidRawAudioStatusError,
    PersistenceConflictError,
    RawAudioNotFoundError,
    RepositoryError,
    SplitRepository,
    _claim_statement,
    _insert_part_statement,
    _validate_schema,
    normalize_database_url,
)


RAW_AUDIO_ID = UUID("12345678-1234-5678-1234-567812345678")


class FakeResult:
    def __init__(
        self,
        *,
        row: Any = None,
        scalar: Any = None,
        rowcount: int = 0,
    ) -> None:
        self.row = row
        self.scalar = scalar
        self.rowcount = rowcount

    def one_or_none(self) -> Any:
        return self.row

    def scalar_one_or_none(self) -> Any:
        return self.scalar


class FakeScalarResult:
    def __init__(self, values: list[UUID]) -> None:
        self._values = values

    def all(self) -> list[UUID]:
        return self._values


class FakeSession:
    def __init__(
        self,
        *,
        execute_results: list[FakeResult | Exception] | None = None,
        scalar_results: list[Any] | None = None,
        scalars_result: list[UUID] | None = None,
    ) -> None:
        self.execute_results = list(execute_results or [])
        self.scalar_results = list(scalar_results or [])
        self.scalars_result = list(scalars_result or [])
        self.statements: list[Any] = []

    def execute(self, statement: Any) -> FakeResult:
        self.statements.append(statement)
        result = self.execute_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def scalar(self, statement: Any) -> Any:
        self.statements.append(statement)
        return self.scalar_results.pop(0)

    def scalars(self, statement: Any) -> FakeScalarResult:
        self.statements.append(statement)
        return FakeScalarResult(self.scalars_result)


class FakeContext(AbstractContextManager[FakeSession]):
    def __init__(self, factory: FakeFactory, session: FakeSession) -> None:
        self.factory = factory
        self.session = session

    def __enter__(self) -> FakeSession:
        return self.session

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.factory.exit_types.append(exc_type)


class FakeFactory:
    def __init__(self, sessions: Iterator[FakeSession]) -> None:
        self._sessions = sessions
        self.exit_types: list[Any] = []

    def begin(self) -> FakeContext:
        return FakeContext(self, next(self._sessions))

    def __call__(self) -> FakeContext:
        return FakeContext(self, next(self._sessions))


def repository_for(session: FakeSession) -> tuple[SplitRepository, FakeFactory]:
    factory = FakeFactory(iter([session]))
    return SplitRepository(factory), factory  # type: ignore[arg-type]


def draft(part_index: int) -> AudioPartDraft:
    start = part_index * 20_000
    end = start + 20_000
    return AudioPartDraft(
        part_index=part_index,
        audio_uri=f"s3://test-bucket/parts/{part_index}.wav",
        lang="en",
        relative_start_ms=start,
        relative_end_ms=end,
        duration_ms=end - start,
    )


def test_claim_statement_is_atomic_and_lock_free() -> None:
    sql = str(
        _claim_statement(RAW_AUDIO_ID).compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "UPDATE raw_audios" in sql
    assert "raw_audios.status IN ('pending', 'failed')" in sql
    assert "RETURNING raw_audios.id, raw_audios.audio_uri, raw_audios.lang" in sql
    assert "FOR UPDATE" not in sql
    assert "advisory" not in sql.lower()


def test_part_insert_uses_unique_key_conflict_skip() -> None:
    sql = str(
        _insert_part_statement(RAW_AUDIO_ID, draft(0)).compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "ON CONFLICT (raw_audio_id, part_index) DO NOTHING" in sql
    assert "RETURNING audio_parts.id" in sql


def test_claim_pending_row_returns_input_contract() -> None:
    session = FakeSession(
        execute_results=[
            FakeResult(
                row=SimpleNamespace(
                    id=RAW_AUDIO_ID,
                    audio_uri="s3://test-bucket/raw.wav",
                    lang="en",
                )
            )
        ]
    )
    repository, _ = repository_for(session)

    claim = repository.claim(RAW_AUDIO_ID)

    assert claim.disposition is ClaimDisposition.CLAIMED
    assert claim.status == "splitting"
    assert claim.audio_uri == "s3://test-bucket/raw.wav"
    assert claim.lang == "en"


@pytest.mark.parametrize(
    ("status", "disposition"),
    [
        ("splitting", ClaimDisposition.ALREADY_PROCESSING),
        ("split_completed", ClaimDisposition.COMPLETED),
    ],
)
def test_unclaimed_known_state_returns_no_op_disposition(
    status: str,
    disposition: ClaimDisposition,
) -> None:
    session = FakeSession(
        execute_results=[FakeResult()],
        scalar_results=[status],
    )
    repository, _ = repository_for(session)

    claim = repository.claim(RAW_AUDIO_ID)

    assert claim.disposition is disposition
    assert claim.status == status
    assert claim.audio_uri is None


def test_missing_or_invalid_state_is_not_guessed() -> None:
    missing, _ = repository_for(
        FakeSession(execute_results=[FakeResult()], scalar_results=[None])
    )
    invalid, _ = repository_for(
        FakeSession(
            execute_results=[FakeResult()],
            scalar_results=["pending_elsewhere"],
        )
    )

    with pytest.raises(RawAudioNotFoundError, match="does not exist"):
        missing.claim(RAW_AUDIO_ID)
    with pytest.raises(InvalidRawAudioStatusError, match="not eligible"):
        invalid.claim(RAW_AUDIO_ID)


def test_persist_inserts_missing_and_reuses_conflicting_part_id() -> None:
    inserted_id = uuid4()
    existing_id = uuid4()
    session = FakeSession(
        execute_results=[
            FakeResult(scalar=inserted_id),
            FakeResult(scalar=None),
            FakeResult(row=SimpleNamespace(id=existing_id, status="pending")),
            FakeResult(rowcount=1),
        ]
    )
    repository, factory = repository_for(session)

    records = repository.persist_parts_and_complete(
        RAW_AUDIO_ID,
        [draft(1), draft(0)],
    )

    assert records[0].part_index == 0
    assert records[0].audio_part_id == inserted_id
    assert records[1].part_index == 1
    assert records[1].audio_part_id == existing_id
    assert records[1].status == "pending"
    assert factory.exit_types == [None]


def test_zero_parts_still_completes_in_one_transaction() -> None:
    session = FakeSession(execute_results=[FakeResult(rowcount=1)])
    repository, factory = repository_for(session)

    assert repository.persist_parts_and_complete(RAW_AUDIO_ID, []) == []
    assert len(session.statements) == 1
    assert factory.exit_types == [None]


def test_completion_precondition_rolls_back_transaction() -> None:
    session = FakeSession(execute_results=[FakeResult(rowcount=0)])
    repository, factory = repository_for(session)

    with pytest.raises(PersistenceConflictError, match="precondition"):
        repository.persist_parts_and_complete(RAW_AUDIO_ID, [])

    assert factory.exit_types == [PersistenceConflictError]


def test_duplicate_draft_indexes_are_rejected_before_transaction() -> None:
    session = FakeSession()
    repository, factory = repository_for(session)

    with pytest.raises(ValueError, match="unique part indexes"):
        repository.persist_parts_and_complete(
            RAW_AUDIO_ID,
            [draft(0), draft(0)],
        )

    assert factory.exit_types == []


def test_pending_parts_are_returned_in_repository_query_order() -> None:
    first = uuid4()
    second = uuid4()
    repository, _ = repository_for(
        FakeSession(scalars_result=[first, second])
    )

    assert repository.list_pending_audio_part_ids(RAW_AUDIO_ID) == [first, second]


def test_audio_part_count_is_returned_as_an_integer() -> None:
    repository, _ = repository_for(FakeSession(scalar_results=[3]))

    assert repository.count_audio_parts(RAW_AUDIO_ID) == 3


def test_mark_failed_updates_one_owned_or_completed_row() -> None:
    session = FakeSession(execute_results=[FakeResult(rowcount=1)])
    repository, factory = repository_for(session)

    repository.mark_failed(
        RAW_AUDIO_ID,
        "split-raw-audio-into-parts upload: unable to persist an audio part.",
    )

    assert len(session.statements) == 1
    assert factory.exit_types == [None]


def test_mark_failed_requires_an_eligible_row() -> None:
    repository, factory = repository_for(
        FakeSession(execute_results=[FakeResult(rowcount=0)])
    )

    with pytest.raises(PersistenceConflictError, match="precondition"):
        repository.mark_failed(RAW_AUDIO_ID, "safe failure")

    assert factory.exit_types == [PersistenceConflictError]


def test_database_exception_text_is_not_exposed() -> None:
    secret = "postgresql://user:secret@example.test/database"
    repository, _ = repository_for(
        FakeSession(execute_results=[SQLAlchemyError(secret)])
    )

    with pytest.raises(RepositoryError) as error:
        repository.claim(RAW_AUDIO_ID)

    assert str(error.value) == "Unable to claim raw audio for splitting."
    assert secret not in str(error.value)


def test_schema_validation_reports_only_missing_contract_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = {
        "raw_audios": [
            "id",
            "status",
            "audio_uri",
            "lang",
            "error",
        ],
        "audio_parts": [
            "id",
            "raw_audio_id",
            "part_index",
            "status",
            "audio_uri",
            "lang",
            "relative_start_ms",
            "relative_end_ms",
            "duration_ms",
            "error",
        ],
    }
    inspector = SimpleNamespace(
        get_columns=lambda table: [
            {"name": column} for column in columns[table]
        ]
    )
    monkeypatch.setattr(repository_module, "inspect", lambda _engine: inspector)

    with pytest.raises(RepositoryError) as error:
        _validate_schema(object())  # type: ignore[arg-type]

    assert str(error.value) == (
        "Database schema is missing required columns "
        "(audio_parts: diarization_uri)."
    )


def test_schema_sqlstate_log_is_safe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "statement with private parameters"

    class MissingColumnError(SQLAlchemyError):
        orig = SimpleNamespace(sqlstate="42703")

    repository, _ = repository_for(
        FakeSession(execute_results=[MissingColumnError(secret)])
    )

    with pytest.raises(RepositoryError, match="Unable to persist"):
        repository.persist_parts_and_complete(RAW_AUDIO_ID, [draft(0)])

    assert "Database schema mismatch" in caplog.text
    assert "sqlstate=42703" in caplog.text
    assert secret not in caplog.text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("postgresql://host/database", "postgresql+psycopg://host/database"),
        ("postgres://host/database", "postgresql+psycopg://host/database"),
        ("postgresql+psycopg://host/database", "postgresql+psycopg://host/database"),
    ],
)
def test_database_url_normalization(source: str, expected: str) -> None:
    assert normalize_database_url(source) == expected
