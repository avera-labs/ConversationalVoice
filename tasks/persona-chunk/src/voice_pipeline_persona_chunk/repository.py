from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from voice_pipeline_models import Chunk


class Disposition(StrEnum):
    CLAIMED = "claimed"
    ALREADY_PROCESSING = "already_processing"
    ALREADY_COMPLETED = "already_completed"
    ALREADY_REJECTED = "already_rejected"
    READY_TO_DISPATCH = "ready_to_dispatch"


@dataclass(frozen=True, slots=True)
class Claim:
    chunk_id: UUID
    disposition: Disposition
    status: str
    audio_part_id: UUID | None = None
    audio_uri: str | None = None
    lang: str | None = None
    duration_ms: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    diarizations: dict | None = None
    separation: dict | None = None
    transcription: dict | None = None
    persona: dict | None = None
    persona_result: dict | None = None


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
            row = session.scalar(
                select(Chunk).where(Chunk.id == identifier).with_for_update()
            )
            if row is None:
                raise RuntimeError("chunk_not_found")
            results = dict(row.final_results or {})
            separation = results.get("separation")
            transcription = results.get("transcription")
            persona_result = results.get("persona")
            persona = row.persona
            if row.status == "persona_generating":
                return Claim(identifier, Disposition.ALREADY_PROCESSING, row.status)
            if row.status in {"extending", "completed"}:
                if not isinstance(persona, dict) or not isinstance(
                    persona_result, dict
                ):
                    raise RuntimeError("invalid_completed_persona_state")
                return self._claim(
                    row,
                    Disposition.ALREADY_COMPLETED,
                    separation,
                    transcription,
                    persona,
                    persona_result,
                )
            if row.status == "rejected":
                if isinstance(persona, dict) and isinstance(persona_result, dict):
                    return self._claim(
                        row,
                        Disposition.ALREADY_COMPLETED,
                        separation,
                        transcription,
                        persona,
                        persona_result,
                    )
                if persona is not None or persona_result is not None:
                    raise RuntimeError("invalid_partial_persona_state")
                return Claim(identifier, Disposition.ALREADY_REJECTED, row.status)
            if (
                row.status in {"persona_generated", "failed"}
                and isinstance(persona, dict)
                and isinstance(persona_result, dict)
            ):
                return self._claim(
                    row,
                    Disposition.READY_TO_DISPATCH,
                    separation,
                    transcription,
                    persona,
                    persona_result,
                )
            if persona is not None or persona_result is not None:
                raise RuntimeError("invalid_partial_persona_state")
            if (
                row.status not in {"transcribed", "failed"}
                or not isinstance(separation, dict)
                or not isinstance(transcription, dict)
                or not isinstance(row.diarizations, dict)
            ):
                raise RuntimeError("invalid_chunk_state")
            row.status = "persona_generating"
            row.error = None
            return self._claim(
                row, Disposition.CLAIMED, separation, transcription, None, None
            )

    @staticmethod
    def _claim(row, disposition, separation, transcription, persona, persona_result):
        return Claim(
            row.id,
            disposition,
            row.status,
            row.audio_part_id,
            row.audio_uri,
            row.lang,
            row.duration_ms,
            row.relative_start_ms,
            row.relative_end_ms,
            dict(row.diarizations) if isinstance(row.diarizations, dict) else None,
            separation if isinstance(separation, dict) else None,
            transcription if isinstance(transcription, dict) else None,
            dict(persona) if isinstance(persona, dict) else None,
            persona_result if isinstance(persona_result, dict) else None,
        )

    def complete(self, claim: Claim, persona: dict, result: dict):
        with self.factory.begin() as session:
            row = session.get(Chunk, claim.chunk_id, with_for_update=True)
            current = dict(row.final_results or {}) if row is not None else {}
            if (
                row is None
                or row.status != "persona_generating"
                or row.audio_part_id != claim.audio_part_id
                or row.audio_uri != claim.audio_uri
                or row.lang != claim.lang
                or row.duration_ms != claim.duration_ms
                or row.relative_start_ms != claim.start_ms
                or row.relative_end_ms != claim.end_ms
                or row.diarizations != claim.diarizations
                or current.get("separation") != claim.separation
                or current.get("transcription") != claim.transcription
            ):
                raise RuntimeError("completion_conflict")
            existing_persona = row.persona
            existing_result = current.get("persona")
            if (existing_persona is not None and existing_persona != persona) or (
                existing_result is not None and existing_result != result
            ):
                raise RuntimeError("persona_conflict")
            row.persona = persona
            current["persona"] = result
            row.final_results = current
            row.status = "persona_generated"
            row.error = None

    def fail_publication(self, identifier: UUID, error: str):
        with self.factory.begin() as session:
            row = session.get(Chunk, identifier, with_for_update=True)
            if row is None:
                raise RuntimeError("state_transition_conflict")
            if row.status == "failed":
                return
            if row.status != "persona_generated":
                raise RuntimeError("state_transition_conflict")
            row.status = "failed"
            row.error = error

    def fail(self, identifier: UUID, error: str):
        with self.factory.begin() as session:
            result = session.execute(
                update(Chunk)
                .where(Chunk.id == identifier, Chunk.status == "persona_generating")
                .values(status="failed", error=error)
            )
            if result.rowcount != 1:
                raise RuntimeError("state_transition_conflict")

    def close(self):
        if self.engine:
            self.engine.dispose()
