"""Short transactions for quality-filter claim, completion, and failure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from sqlalchemy import create_engine, inspect, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from voice_pipeline_models import AudioPart, Chunk

from .config import EnvironmentSettings


class RepositoryError(RuntimeError):
    pass


class AudioPartNotFoundError(RepositoryError):
    pass


class InvalidAudioPartStatusError(RepositoryError):
    pass


class PersistenceConflictError(RepositoryError):
    pass


class ClaimDisposition(StrEnum):
    CLAIMED = "claimed"
    ALREADY_PROCESSING = "already_processing"
    ALREADY_COMPLETED = "already_completed"
    NOT_READY = "not_ready"


@dataclass(frozen=True, slots=True)
class AudioPartClaim:
    audio_part_id: UUID
    disposition: ClaimDisposition
    status: str
    audio_uri: str | None = None
    diarization_uri: str | None = None
    duration_ms: int | None = None
    lang: str | None = None


@dataclass(frozen=True, slots=True)
class PersistedChunk:
    chunk_id: UUID
    chunk_index: int
    audio_uri: str
    lang: str
    duration_ms: int
    start_ms: int
    end_ms: int


def _validate_planned_chunks(
    claim: AudioPartClaim, chunks: tuple[PersistedChunk, ...]
) -> None:
    if claim.duration_ms is None or claim.duration_ms <= 0:
        raise PersistenceConflictError("Audio part duration is invalid.")
    previous_end = 0
    for expected_index, chunk in enumerate(chunks):
        if (
            chunk.chunk_index != expected_index
            or not chunk.audio_uri
            or not chunk.lang
            or chunk.lang != claim.lang
            or chunk.start_ms < previous_end
            or chunk.start_ms < 0
            or chunk.end_ms > claim.duration_ms
            or chunk.duration_ms <= 0
            or chunk.duration_ms != chunk.end_ms - chunk.start_ms
        ):
            raise PersistenceConflictError("Planned chunks are invalid.")
        previous_end = chunk.end_ms


def normalize_database_url(value: str) -> str:
    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg://", 1)
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg://", 1)
    return value


def _validate_schema(engine: Engine) -> None:
    audio_part_required = {"id", "status", "audio_uri", "diarization_uri", "duration_ms", "lang", "error"}
    chunk_required = {
        "id", "audio_part_id", "chunk_index", "status", "audio_uri", "lang",
        "duration_ms", "relative_start_ms", "relative_end_ms", "diarization_model",
        "diarizations", "persona", "final_results", "error",
    }
    audio_part_columns = {item["name"] for item in inspect(engine).get_columns("audio_parts")}
    chunk_columns = {item["name"] for item in inspect(engine).get_columns("chunks")}
    if audio_part_required - audio_part_columns or chunk_required - chunk_columns:
        raise RepositoryError("Database schema is missing required quality-filter columns.")


def _claim_statement(audio_part_id: UUID):
    return (
        update(AudioPart)
        .where(AudioPart.id == audio_part_id, AudioPart.status.in_(("diarized", "failed")))
        .values(status="filtering", error=None)
        .returning(
            AudioPart.id,
            AudioPart.audio_uri,
            AudioPart.diarization_uri,
            AudioPart.duration_ms,
            AudioPart.lang,
        )
    )


class QualityFilterRepository:
    def __init__(
        self, session_factory: sessionmaker[Session], *, engine: Engine | None = None
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine

    @classmethod
    def create(cls, settings: EnvironmentSettings) -> QualityFilterRepository:
        engine: Engine | None = None
        try:
            engine = create_engine(normalize_database_url(settings.database_url), pool_pre_ping=True)
            _validate_schema(engine)
        except RepositoryError:
            if engine is not None:
                engine.dispose()
            raise
        except SQLAlchemyError as exc:
            if engine is not None:
                engine.dispose()
            raise RepositoryError("Unable to configure database access.") from exc
        return cls(sessionmaker(bind=engine, expire_on_commit=False), engine=engine)

    def claim(self, audio_part_id: UUID) -> AudioPartClaim:
        try:
            with self._session_factory.begin() as session:
                claimed = session.execute(_claim_statement(audio_part_id)).one_or_none()
                if claimed is not None:
                    return AudioPartClaim(
                        claimed.id,
                        ClaimDisposition.CLAIMED,
                        "filtering",
                        claimed.audio_uri,
                        claimed.diarization_uri,
                        claimed.duration_ms,
                        claimed.lang,
                    )
                status = session.scalar(select(AudioPart.status).where(AudioPart.id == audio_part_id))
                if status is None:
                    raise AudioPartNotFoundError("Audio part does not exist.")
                if status in {"filtering", "diarizing"}:
                    disposition = ClaimDisposition.ALREADY_PROCESSING
                elif status == "completed":
                    disposition = ClaimDisposition.ALREADY_COMPLETED
                elif status == "pending":
                    disposition = ClaimDisposition.NOT_READY
                else:
                    raise InvalidAudioPartStatusError("Audio part status is invalid.")
                return AudioPartClaim(audio_part_id, disposition, status)
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to claim audio part.") from exc

    def complete(
        self, claim: AudioPartClaim, chunks: tuple[PersistedChunk, ...]
    ) -> tuple[UUID, ...]:
        _validate_planned_chunks(claim, chunks)
        try:
            with self._session_factory.begin() as session:
                part = session.scalar(
                    select(AudioPart).where(AudioPart.id == claim.audio_part_id).with_for_update()
                )
                if part is None:
                    raise AudioPartNotFoundError("Audio part does not exist.")
                if (
                    part.status != "filtering"
                    or part.audio_uri != claim.audio_uri
                    or part.diarization_uri != claim.diarization_uri
                    or part.duration_ms != claim.duration_ms
                    or part.lang != claim.lang
                ):
                    raise PersistenceConflictError("Audio part completion precondition was not met.")
                existing = tuple(
                    session.scalars(
                        select(Chunk)
                        .where(Chunk.audio_part_id == claim.audio_part_id)
                        .order_by(Chunk.chunk_index)
                        .with_for_update()
                    )
                )
                if existing:
                    if len(existing) != len(chunks):
                        raise PersistenceConflictError("Existing chunks do not match the planned result.")
                    for row, planned in zip(existing, chunks, strict=True):
                        owned_matches = (
                            row.chunk_index == planned.chunk_index
                            and row.status == "pending"
                            and row.audio_uri == planned.audio_uri
                            and row.lang == planned.lang
                            and row.duration_ms == planned.duration_ms
                            and row.relative_start_ms == planned.start_ms
                            and row.relative_end_ms == planned.end_ms
                            and row.error is None
                        )
                        downstream_empty = all(
                            value is None
                            for value in (
                                row.diarization_model,
                                row.diarizations,
                                row.persona,
                                row.final_results,
                            )
                        )
                        if not owned_matches or not downstream_empty:
                            raise PersistenceConflictError("Existing chunk fields conflict with the planned result.")
                    chunk_ids = tuple(row.id for row in existing)
                else:
                    for planned in chunks:
                        session.add(
                            Chunk(
                                id=planned.chunk_id,
                                audio_part_id=claim.audio_part_id,
                                chunk_index=planned.chunk_index,
                                status="pending",
                                audio_uri=planned.audio_uri,
                                lang=planned.lang,
                                duration_ms=planned.duration_ms,
                                relative_start_ms=planned.start_ms,
                                relative_end_ms=planned.end_ms,
                                error=None,
                            )
                        )
                    chunk_ids = tuple(planned.chunk_id for planned in chunks)
                part.status = "completed"
                part.error = None
                return chunk_ids
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to complete quality filtering.") from exc

    def mark_failed(self, audio_part_id: UUID, error: str) -> None:
        try:
            with self._session_factory.begin() as session:
                result = session.execute(
                    update(AudioPart)
                    .where(AudioPart.id == audio_part_id, AudioPart.status == "filtering")
                    .values(status="failed", error=error)
                )
                if result.rowcount != 1:
                    raise PersistenceConflictError("Audio part failure precondition was not met.")
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to mark quality filtering as failed.") from exc

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
