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
    ALREADY_TRANSCRIBED = "already_transcribed"
    ALREADY_REJECTED = "already_rejected"


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
            if row.status == "transcribing":
                return Claim(identifier, Disposition.ALREADY_PROCESSING, row.status)
            if row.status == "transcribed" or transcription is not None:
                if row.status not in {
                    "transcribed",
                    "persona_generating",
                    "persona_generated",
                    "extending",
                    "completed",
                    "rejected",
                    "failed",
                } or not isinstance(transcription, dict):
                    raise RuntimeError("invalid_transcription_state")
                return Claim(
                    row.id,
                    Disposition.ALREADY_TRANSCRIBED,
                    row.status,
                    row.audio_part_id,
                    row.audio_uri,
                    row.lang,
                    row.duration_ms,
                    row.relative_start_ms,
                    row.relative_end_ms,
                    dict(row.diarizations)
                    if isinstance(row.diarizations, dict)
                    else None,
                    separation if isinstance(separation, dict) else None,
                    transcription,
                )
            if row.status == "rejected":
                return Claim(identifier, Disposition.ALREADY_REJECTED, row.status)
            if row.status not in {"separated", "failed"} or not isinstance(
                separation, dict
            ):
                raise RuntimeError("invalid_chunk_state")
            if not isinstance(row.diarizations, dict):
                raise TypeError("missing_chunk_diarization")
            row.status = "transcribing"
            row.error = None
            return Claim(
                row.id,
                Disposition.CLAIMED,
                "transcribing",
                row.audio_part_id,
                row.audio_uri,
                row.lang,
                row.duration_ms,
                row.relative_start_ms,
                row.relative_end_ms,
                dict(row.diarizations),
                separation,
            )

    def complete(self, claim: Claim, result: dict):
        with self.factory.begin() as session:
            row = session.get(Chunk, claim.chunk_id, with_for_update=True)
            current = dict(row.final_results or {}) if row is not None else {}
            if (
                row is None
                or row.status != "transcribing"
                or row.audio_part_id != claim.audio_part_id
                or row.audio_uri != claim.audio_uri
                or row.lang != claim.lang
                or row.duration_ms != claim.duration_ms
                or row.relative_start_ms != claim.start_ms
                or row.relative_end_ms != claim.end_ms
                or row.diarizations != claim.diarizations
                or current.get("separation") != claim.separation
            ):
                raise RuntimeError("completion_conflict")
            existing = current.get("transcription")
            if existing is not None and existing != result:
                raise RuntimeError("transcription_conflict")
            current["transcription"] = result
            row.final_results = current
            row.status = "transcribed"
            row.error = None

    def fail(self, identifier: UUID, error: str):
        with self.factory.begin() as session:
            result = session.execute(
                update(Chunk)
                .where(Chunk.id == identifier, Chunk.status == "transcribing")
                .values(status="failed", error=error)
            )
            if result.rowcount != 1:
                raise RuntimeError("state_transition_conflict")

    def close(self):
        if self.engine:
            self.engine.dispose()
