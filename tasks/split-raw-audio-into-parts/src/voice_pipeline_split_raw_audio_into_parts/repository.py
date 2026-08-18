from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import logging
from uuid import UUID, uuid4

from sqlalchemy import create_engine, func, inspect, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from voice_pipeline_models import AudioPart, RawAudio

from .config import EnvironmentSettings


logger = logging.getLogger(__name__)
_REQUIRED_TABLE_COLUMNS = {
    "raw_audios": frozenset({"id", "status", "audio_uri", "lang", "error"}),
    "audio_parts": frozenset(
        {
            "id",
            "raw_audio_id",
            "part_index",
            "status",
            "audio_uri",
            "diarization_uri",
            "lang",
            "relative_start_ms",
            "relative_end_ms",
            "duration_ms",
            "error",
        }
    ),
}
_SCHEMA_SQLSTATES = frozenset({"42703", "42P01"})


class RepositoryError(RuntimeError):
    """Raised when a persistence operation cannot complete safely."""


class RawAudioNotFoundError(RepositoryError):
    """Raised when the task identifier does not resolve to a raw audio row."""


class InvalidRawAudioStatusError(RepositoryError):
    """Raised when a raw audio row is outside this task's state machine."""


class PersistenceConflictError(RepositoryError):
    """Raised when a required state transition loses its precondition."""


class ClaimDisposition(StrEnum):
    """Result of the atomic task-start status transition."""

    CLAIMED = "claimed"
    ALREADY_PROCESSING = "already_processing"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class RawAudioClaim:
    """Detached result of attempting to claim one raw audio row."""

    raw_audio_id: UUID
    disposition: ClaimDisposition
    status: str
    audio_uri: str | None = None
    lang: str | None = None


@dataclass(frozen=True, slots=True)
class AudioPartDraft:
    """Persistence values for one deterministically indexed audio part."""

    part_index: int
    audio_uri: str
    lang: str
    relative_start_ms: int
    relative_end_ms: int
    duration_ms: int

    def __post_init__(self) -> None:
        if self.part_index < 0:
            raise ValueError("part_index must not be negative")
        if not self.audio_uri:
            raise ValueError("audio_uri must not be empty")
        if not self.lang:
            raise ValueError("lang must not be empty")
        if self.relative_start_ms < 0:
            raise ValueError("relative_start_ms must not be negative")
        if self.relative_end_ms <= self.relative_start_ms:
            raise ValueError("relative_end_ms must be greater than relative_start_ms")
        if self.duration_ms != self.relative_end_ms - self.relative_start_ms:
            raise ValueError("duration_ms must match the persisted boundaries")


@dataclass(frozen=True, slots=True)
class PersistedAudioPart:
    """Identity and current status of an inserted or reused audio part."""

    audio_part_id: UUID
    part_index: int
    status: str


def normalize_database_url(database_url: str) -> str:
    """Select the installed psycopg 3 driver for generic PostgreSQL URLs."""

    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


def _validate_schema(engine: Engine) -> None:
    """Fail before task consumption when required persistence fields are absent."""

    inspector = inspect(engine)
    missing_by_table: dict[str, list[str]] = {}
    for table_name, required_columns in _REQUIRED_TABLE_COLUMNS.items():
        observed_columns = {
            column["name"] for column in inspector.get_columns(table_name)
        }
        missing = sorted(required_columns - observed_columns)
        if missing:
            missing_by_table[table_name] = missing

    if missing_by_table:
        details = "; ".join(
            f"{table}: {', '.join(columns)}"
            for table, columns in sorted(missing_by_table.items())
        )
        raise RepositoryError(
            f"Database schema is missing required columns ({details})."
        )


def _log_database_failure(operation: str, error: SQLAlchemyError) -> None:
    """Log a bounded database category without connection or statement data."""

    original = getattr(error, "orig", None)
    sqlstate = getattr(original, "sqlstate", None)
    if sqlstate in _SCHEMA_SQLSTATES:
        logger.error(
            (
                "Database schema mismatch operation=%s sqlstate=%s. "
                "Apply the authoritative schema before retrying the task."
            ),
            operation,
            sqlstate,
        )
        return
    logger.error(
        "Database operation failed operation=%s sqlstate=%s.",
        operation,
        sqlstate or "unavailable",
    )


def _claim_statement(raw_audio_id: UUID):
    return (
        update(RawAudio)
        .where(
            RawAudio.id == raw_audio_id,
            RawAudio.status.in_(("pending", "failed")),
        )
        .values(status="splitting", error=None)
        .returning(RawAudio.id, RawAudio.audio_uri, RawAudio.lang)
    )


def _insert_part_statement(raw_audio_id: UUID, draft: AudioPartDraft):
    return (
        insert(AudioPart)
        .values(
            id=uuid4(),
            raw_audio_id=raw_audio_id,
            part_index=draft.part_index,
            status="pending",
            audio_uri=draft.audio_uri,
            diarization_uri=None,
            lang=draft.lang,
            relative_start_ms=draft.relative_start_ms,
            relative_end_ms=draft.relative_end_ms,
            duration_ms=draft.duration_ms,
            error=None,
        )
        .on_conflict_do_nothing(
            index_elements=(AudioPart.raw_audio_id, AudioPart.part_index)
        )
        .returning(AudioPart.id)
    )


class SplitRepository:
    """Short-lived transactions for task state and audio-part persistence."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        engine: Engine | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine

    @classmethod
    def create(cls, settings: EnvironmentSettings) -> SplitRepository:
        engine: Engine | None = None
        try:
            engine = create_engine(
                normalize_database_url(settings.database_url),
                pool_pre_ping=True,
            )
            _validate_schema(engine)
        except RepositoryError:
            if engine is not None:
                engine.dispose()
            raise
        except SQLAlchemyError as exc:
            if engine is not None:
                engine.dispose()
            raise RepositoryError("Unable to configure database access.") from exc
        return cls(
            sessionmaker(bind=engine, expire_on_commit=False),
            engine=engine,
        )

    def claim(self, raw_audio_id: UUID) -> RawAudioClaim:
        """Atomically change pending or failed to splitting without a lock."""

        try:
            with self._session_factory.begin() as session:
                claimed = session.execute(
                    _claim_statement(raw_audio_id)
                ).one_or_none()
                if claimed is not None:
                    return RawAudioClaim(
                        raw_audio_id=claimed.id,
                        disposition=ClaimDisposition.CLAIMED,
                        status="splitting",
                        audio_uri=claimed.audio_uri,
                        lang=claimed.lang,
                    )

                current_status = session.scalar(
                    select(RawAudio.status).where(RawAudio.id == raw_audio_id)
                )
                if current_status is None:
                    raise RawAudioNotFoundError("Raw audio does not exist.")
                if current_status == "splitting":
                    return RawAudioClaim(
                        raw_audio_id=raw_audio_id,
                        disposition=ClaimDisposition.ALREADY_PROCESSING,
                        status=current_status,
                    )
                if current_status == "split_completed":
                    return RawAudioClaim(
                        raw_audio_id=raw_audio_id,
                        disposition=ClaimDisposition.COMPLETED,
                        status=current_status,
                    )
                raise InvalidRawAudioStatusError(
                    "Raw audio is not eligible for splitting."
                )
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to claim raw audio for splitting.") from exc

    def persist_parts_and_complete(
        self,
        raw_audio_id: UUID,
        drafts: list[AudioPartDraft],
    ) -> list[PersistedAudioPart]:
        """Insert missing parts and complete the raw audio in one transaction."""

        part_indexes = [draft.part_index for draft in drafts]
        if len(part_indexes) != len(set(part_indexes)):
            raise ValueError("audio part drafts must use unique part indexes")

        persisted: list[PersistedAudioPart] = []
        try:
            with self._session_factory.begin() as session:
                for draft in sorted(drafts, key=lambda value: value.part_index):
                    inserted_id = session.execute(
                        _insert_part_statement(raw_audio_id, draft)
                    ).scalar_one_or_none()
                    if inserted_id is not None:
                        persisted.append(
                            PersistedAudioPart(
                                audio_part_id=inserted_id,
                                part_index=draft.part_index,
                                status="pending",
                            )
                        )
                        continue

                    existing = session.execute(
                        select(AudioPart.id, AudioPart.status).where(
                            AudioPart.raw_audio_id == raw_audio_id,
                            AudioPart.part_index == draft.part_index,
                        )
                    ).one_or_none()
                    if existing is None:
                        raise PersistenceConflictError(
                            "Conflicting audio part could not be reused."
                        )
                    persisted.append(
                        PersistedAudioPart(
                            audio_part_id=existing.id,
                            part_index=draft.part_index,
                            status=existing.status,
                        )
                    )

                completion = session.execute(
                    update(RawAudio)
                    .where(
                        RawAudio.id == raw_audio_id,
                        RawAudio.status == "splitting",
                    )
                    .values(status="split_completed", error=None)
                )
                if completion.rowcount != 1:
                    raise PersistenceConflictError(
                        "Raw audio completion precondition was not met."
                    )
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            _log_database_failure("persist_parts_and_complete", exc)
            raise RepositoryError("Unable to persist audio parts.") from exc
        return persisted

    def list_pending_audio_part_ids(self, raw_audio_id: UUID) -> list[UUID]:
        try:
            with self._session_factory() as session:
                return list(
                    session.scalars(
                        select(AudioPart.id)
                        .where(
                            AudioPart.raw_audio_id == raw_audio_id,
                            AudioPart.status == "pending",
                        )
                        .order_by(AudioPart.part_index)
                    ).all()
                )
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to query pending audio parts.") from exc

    def count_audio_parts(self, raw_audio_id: UUID) -> int:
        try:
            with self._session_factory() as session:
                return int(
                    session.scalar(
                        select(func.count())
                        .select_from(AudioPart)
                        .where(AudioPart.raw_audio_id == raw_audio_id)
                    )
                    or 0
                )
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to count audio parts.") from exc

    def mark_failed(self, raw_audio_id: UUID, error: str) -> None:
        try:
            with self._session_factory.begin() as session:
                result = session.execute(
                    update(RawAudio)
                    .where(
                        RawAudio.id == raw_audio_id,
                        RawAudio.status.in_(("splitting", "split_completed", "failed")),
                    )
                    .values(status="failed", error=error)
                )
                if result.rowcount != 1:
                    raise PersistenceConflictError(
                        "Raw audio failure precondition was not met."
                    )
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to mark raw audio as failed.") from exc

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
