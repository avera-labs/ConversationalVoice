"""Short database transactions for the diarization state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from sqlalchemy import create_engine, inspect, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from voice_pipeline_models import AudioPart

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
    DISPATCH_READY = "dispatch_ready"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class AudioPartClaim:
    audio_part_id: UUID
    disposition: ClaimDisposition
    status: str
    audio_uri: str | None = None
    duration_ms: int | None = None


def normalize_database_url(value: str) -> str:
    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg://", 1)
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg://", 1)
    return value


def _validate_schema(engine: Engine) -> None:
    required = {"id", "status", "audio_uri", "duration_ms", "diarization_uri", "error"}
    observed = {column["name"] for column in inspect(engine).get_columns("audio_parts")}
    if required - observed:
        raise RepositoryError("Database schema is missing required audio part columns.")


def _claim_statement(audio_part_id: UUID):
    return (
        update(AudioPart)
        .where(
            AudioPart.id == audio_part_id, AudioPart.status.in_(("pending", "failed"))
        )
        .values(status="diarizing", error=None, diarization_uri=None)
        .returning(AudioPart.id, AudioPart.audio_uri, AudioPart.duration_ms)
    )


class DiarizationRepository:
    def __init__(
        self, session_factory: sessionmaker[Session], *, engine: Engine | None = None
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine

    @classmethod
    def create(cls, settings: EnvironmentSettings) -> DiarizationRepository:
        engine: Engine | None = None
        try:
            engine = create_engine(
                normalize_database_url(settings.database_url), pool_pre_ping=True
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
        return cls(sessionmaker(bind=engine, expire_on_commit=False), engine=engine)

    def claim(self, audio_part_id: UUID) -> AudioPartClaim:
        try:
            with self._session_factory.begin() as session:
                claimed = session.execute(_claim_statement(audio_part_id)).one_or_none()
                if claimed is not None:
                    return AudioPartClaim(
                        audio_part_id=claimed.id,
                        disposition=ClaimDisposition.CLAIMED,
                        status="diarizing",
                        audio_uri=claimed.audio_uri,
                        duration_ms=claimed.duration_ms,
                    )
                status = session.scalar(
                    select(AudioPart.status).where(AudioPart.id == audio_part_id)
                )
                if status is None:
                    raise AudioPartNotFoundError("Audio part does not exist.")
                if status in {"diarizing", "filtering"}:
                    disposition = ClaimDisposition.ALREADY_PROCESSING
                elif status == "diarized":
                    disposition = ClaimDisposition.DISPATCH_READY
                elif status == "completed":
                    disposition = ClaimDisposition.COMPLETED
                else:
                    raise InvalidAudioPartStatusError("Audio part status is invalid.")
                return AudioPartClaim(audio_part_id, disposition, status)
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to claim audio part.") from exc

    def complete(self, audio_part_id: UUID, diarization_uri: str) -> None:
        try:
            with self._session_factory.begin() as session:
                result = session.execute(
                    update(AudioPart)
                    .where(
                        AudioPart.id == audio_part_id, AudioPart.status == "diarizing"
                    )
                    .values(
                        status="diarized", diarization_uri=diarization_uri, error=None
                    )
                )
                if result.rowcount != 1:
                    raise PersistenceConflictError(
                        "Audio part completion precondition was not met."
                    )
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to complete audio part.") from exc

    def mark_processing_failed(self, audio_part_id: UUID, error: str) -> None:
        self._mark_failed(audio_part_id, expected_status="diarizing", error=error)

    def mark_dispatch_failed(self, audio_part_id: UUID, error: str) -> None:
        self._mark_failed(audio_part_id, expected_status="diarized", error=error)

    def _mark_failed(
        self, audio_part_id: UUID, *, expected_status: str, error: str
    ) -> None:
        try:
            with self._session_factory.begin() as session:
                result = session.execute(
                    update(AudioPart)
                    .where(
                        AudioPart.id == audio_part_id,
                        AudioPart.status == expected_status,
                    )
                    .values(status="failed", error=error)
                )
                if result.rowcount != 1:
                    raise PersistenceConflictError(
                        "Audio part failure precondition was not met."
                    )
        except RepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise RepositoryError("Unable to mark audio part as failed.") from exc

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
