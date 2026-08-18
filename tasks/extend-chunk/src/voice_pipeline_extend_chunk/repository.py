from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from voice_pipeline_models import AudioPart, Chunk


class Disposition(StrEnum):
    CLAIMED = "claimed"
    ALREADY_PROCESSING = "already_processing"
    ALREADY_COMPLETED = "already_completed"
    ALREADY_REJECTED = "already_rejected"


@dataclass(frozen=True, slots=True)
class Claim:
    chunk_id: UUID
    disposition: Disposition
    status: str
    audio_part_id: UUID | None = None
    chunk_audio_uri: str | None = None
    audio_part_audio_uri: str | None = None
    lang: str | None = None
    duration_ms: int | None = None
    diarizations: dict | None = None
    separation: dict | None = None
    transcription: dict | None = None
    persona: dict | None = None
    persona_result: dict | None = None
    extension_result: dict | None = None


def normalize_url(value: str) -> str:
    return (
        value.replace("postgresql://", "postgresql+psycopg://", 1)
        if value.startswith("postgresql://")
        else value
    )


class Repository:
    def __init__(self, factory, engine=None):
        self.factory = factory
        self.engine = engine

    @classmethod
    def create(cls, environment):
        engine = create_engine(
            normalize_url(environment.database_url), pool_pre_ping=True
        )
        return cls(sessionmaker(bind=engine, expire_on_commit=False), engine)

    def claim(self, identifier: UUID) -> Claim:
        with self.factory.begin() as session:
            result = session.execute(
                select(Chunk, AudioPart.audio_uri)
                .join(AudioPart, AudioPart.id == Chunk.audio_part_id)
                .where(Chunk.id == identifier)
                .with_for_update(of=Chunk)
            ).one_or_none()
            if result is None:
                raise RuntimeError("chunk_not_found")
            row, audio_part_audio_uri = result
            results = dict(row.final_results or {})
            separation = results.get("separation")
            transcription = results.get("transcription")
            persona_result = results.get("persona")
            extension_result = results.get("dialogue_extension")
            if row.status == "extending":
                return Claim(identifier, Disposition.ALREADY_PROCESSING, row.status)
            if row.status == "rejected":
                return Claim(identifier, Disposition.ALREADY_REJECTED, row.status)
            if row.status == "completed":
                if not isinstance(extension_result, dict):
                    raise RuntimeError("invalid_completed_extension_state")
                return self._claim(
                    row,
                    audio_part_audio_uri,
                    Disposition.ALREADY_COMPLETED,
                    separation,
                    transcription,
                    persona_result,
                    extension_result,
                )
            if extension_result is not None:
                raise RuntimeError("invalid_partial_extension_state")
            if (
                row.status not in {"persona_generated", "failed"}
                or not isinstance(separation, dict)
                or not isinstance(transcription, dict)
                or not isinstance(row.persona, dict)
                or not isinstance(persona_result, dict)
            ):
                raise RuntimeError("invalid_chunk_state")
            row.status = "extending"
            row.error = None
            return self._claim(
                row,
                audio_part_audio_uri,
                Disposition.CLAIMED,
                separation,
                transcription,
                persona_result,
                None,
            )

    @staticmethod
    def _claim(
        row,
        audio_part_audio_uri,
        disposition,
        separation,
        transcription,
        persona_result,
        extension_result,
    ):
        return Claim(
            row.id,
            disposition,
            row.status,
            row.audio_part_id,
            row.audio_uri,
            audio_part_audio_uri,
            row.lang,
            row.duration_ms,
            dict(row.diarizations) if isinstance(row.diarizations, dict) else None,
            separation if isinstance(separation, dict) else None,
            transcription if isinstance(transcription, dict) else None,
            dict(row.persona) if isinstance(row.persona, dict) else None,
            persona_result if isinstance(persona_result, dict) else None,
            extension_result if isinstance(extension_result, dict) else None,
        )

    def complete(self, claim: Claim, result: dict):
        with self.factory.begin() as session:
            selected = session.execute(
                select(Chunk, AudioPart.audio_uri)
                .join(AudioPart, AudioPart.id == Chunk.audio_part_id)
                .where(Chunk.id == claim.chunk_id)
                .with_for_update(of=Chunk)
            ).one_or_none()
            if selected is None:
                raise RuntimeError("completion_conflict")
            row, audio_part_audio_uri = selected
            current = dict(row.final_results or {})
            if (
                row.status != "extending"
                or row.audio_part_id != claim.audio_part_id
                or row.audio_uri != claim.chunk_audio_uri
                or audio_part_audio_uri != claim.audio_part_audio_uri
                or row.lang != claim.lang
                or row.duration_ms != claim.duration_ms
                or current.get("separation") != claim.separation
                or current.get("transcription") != claim.transcription
                or row.persona != claim.persona
                or current.get("persona") != claim.persona_result
            ):
                raise RuntimeError("completion_conflict")
            existing = current.get("dialogue_extension")
            if existing is not None and existing != result:
                raise RuntimeError("dialogue_extension_conflict")
            current["dialogue_extension"] = result
            row.final_results = current
            row.status = "completed"
            row.error = None

    def reject(self, identifier: UUID, error: str):
        self._finish(identifier, "rejected", error)

    def fail(self, identifier: UUID, error: str):
        self._finish(identifier, "failed", error)

    def _finish(self, identifier: UUID, status: str, error: str):
        with self.factory.begin() as session:
            result = session.execute(
                update(Chunk)
                .where(Chunk.id == identifier, Chunk.status == "extending")
                .values(status=status, error=error)
            )
            if result.rowcount != 1:
                raise RuntimeError("state_transition_conflict")

    def close(self):
        if self.engine:
            self.engine.dispose()
