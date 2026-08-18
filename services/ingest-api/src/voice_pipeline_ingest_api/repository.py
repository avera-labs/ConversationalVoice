from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from voice_pipeline_models import RawAudio


class RawAudioRepositoryError(RuntimeError):
    """Raised when a raw audio persistence operation fails."""


class DuplicateContentError(RawAudioRepositoryError):
    """Raised when another request has already inserted the same content."""

    def __init__(self, record: RawAudioRecord) -> None:
        super().__init__("Raw audio content already exists.")
        self.record = record


@dataclass(frozen=True, slots=True)
class RawAudioRecord:
    """Detached representation of a raw_audios row."""

    id: UUID
    status: str
    audio_uri: str | None
    content_sha1: str | None
    title: str | None
    source_url: str | None
    lang: str
    meta: dict[str, Any]
    duration_ms: int | None
    size_bytes: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, model: RawAudio) -> RawAudioRecord:
        return cls(
            id=model.id,
            status=model.status,
            audio_uri=model.audio_uri,
            content_sha1=model.content_sha1,
            title=model.title,
            source_url=model.source_url,
            lang=model.lang,
            meta=dict(model.meta),
            duration_ms=model.duration_ms,
            size_bytes=model.size_bytes,
            error=model.error,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


@dataclass(frozen=True, slots=True)
class RawAudioCreate:
    """Values required to create a normalized raw audio row."""

    id: UUID
    audio_uri: str
    content_sha1: str
    title: str | None
    source_url: str | None
    lang: str
    meta: dict[str, Any]
    duration_ms: int
    size_bytes: int


class RawAudioRepository:
    """Short-lived transactional access to raw_audios."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def find_by_content_sha1(self, content_sha1: str) -> RawAudioRecord | None:
        try:
            with self._session_factory() as session:
                model = session.scalar(
                    select(RawAudio).where(RawAudio.content_sha1 == content_sha1)
                )
                return None if model is None else RawAudioRecord.from_model(model)
        except SQLAlchemyError as exc:
            raise RawAudioRepositoryError("Unable to query raw audio content.") from exc

    def get(self, raw_audio_id: UUID) -> RawAudioRecord | None:
        try:
            with self._session_factory() as session:
                model = session.get(RawAudio, raw_audio_id)
                return None if model is None else RawAudioRecord.from_model(model)
        except SQLAlchemyError as exc:
            raise RawAudioRepositoryError("Unable to query raw audio.") from exc

    def create(self, values: RawAudioCreate) -> RawAudioRecord:
        model = RawAudio(
            id=values.id,
            status="pending",
            audio_uri=values.audio_uri,
            content_sha1=values.content_sha1,
            title=values.title,
            source_url=values.source_url,
            lang=values.lang,
            meta=values.meta,
            duration_ms=values.duration_ms,
            size_bytes=values.size_bytes,
            error=None,
        )
        try:
            with self._session_factory.begin() as session:
                session.add(model)
                session.flush()
                record = RawAudioRecord.from_model(model)
            return record
        except IntegrityError as exc:
            existing = self.find_by_content_sha1(values.content_sha1)
            if existing is not None:
                raise DuplicateContentError(existing) from exc
            raise RawAudioRepositoryError("Unable to create raw audio.") from exc
        except SQLAlchemyError as exc:
            raise RawAudioRepositoryError("Unable to create raw audio.") from exc

    def mark_failed(self, raw_audio_id: UUID, error: str) -> None:
        try:
            with self._session_factory.begin() as session:
                result = session.execute(
                    update(RawAudio)
                    .where(
                        RawAudio.id == raw_audio_id,
                        RawAudio.status == "pending",
                    )
                    .values(status="failed", error=error)
                )
                if result.rowcount != 1:
                    raise RawAudioRepositoryError(
                        "Raw audio is missing or no longer pending."
                    )
        except SQLAlchemyError as exc:
            raise RawAudioRepositoryError(
                "Unable to mark raw audio as failed."
            ) from exc
