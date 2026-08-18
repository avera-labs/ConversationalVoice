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
    ALREADY_SEPARATED = "already_separated"
    ALREADY_REJECTED = "already_rejected"


@dataclass(frozen=True, slots=True)
class Claim:
    chunk_id: UUID
    disposition: Disposition
    status: str
    audio_uri: str | None = None
    duration_ms: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    lang: str | None = None
    part_duration_ms: int | None = None
    diarization_uri: str | None = None
    audio_part_id: UUID | None = None
    diarizations: dict | None = None
    separation: dict | None = None


def normalize_url(value):
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
    def create(cls, env):
        engine = create_engine(normalize_url(env.database_url), pool_pre_ping=True)
        return cls(sessionmaker(bind=engine, expire_on_commit=False), engine)

    def claim(self, identifier):
        claimed = None
        contract_drift = False
        with self.factory.begin() as session:
            row = session.scalar(
                select(Chunk).where(Chunk.id == identifier).with_for_update()
            )
            if row is None:
                raise RuntimeError("chunk_not_found")
            separation = dict(row.final_results or {}).get("separation")
            if separation is not None:
                if row.status not in {
                    "separated",
                    "transcribing",
                    "transcribed",
                    "persona_generating",
                    "persona_generated",
                    "extending",
                    "completed",
                    "rejected",
                    "failed",
                } or not isinstance(separation, dict):
                    raise RuntimeError("invalid_separation_state")
                return Claim(
                    row.id,
                    Disposition.ALREADY_SEPARATED,
                    row.status,
                    row.audio_uri,
                    row.duration_ms,
                    row.relative_start_ms,
                    row.relative_end_ms,
                    row.lang,
                    audio_part_id=row.audio_part_id,
                    diarizations=(
                        dict(row.diarizations)
                        if isinstance(row.diarizations, dict)
                        else None
                    ),
                    separation=separation,
                )
            if row.status in {"pending", "failed"}:
                row.status = "separating"
                row.error = None
                part = session.get(AudioPart, row.audio_part_id)
                contract_drift = (
                    not part
                    or part.status != "completed"
                    or part.lang != row.lang
                    or not part.diarization_uri
                    or not row.audio_uri
                    or row.duration_ms < 20000
                    or row.relative_start_ms < 0
                    or row.relative_start_ms >= row.relative_end_ms
                    or row.relative_end_ms > part.duration_ms
                    or row.duration_ms != row.relative_end_ms - row.relative_start_ms
                )
                claimed = Claim(
                    row.id,
                    Disposition.CLAIMED,
                    "separating",
                    row.audio_uri,
                    row.duration_ms,
                    row.relative_start_ms,
                    row.relative_end_ms,
                    row.lang,
                    part.duration_ms if part else None,
                    part.diarization_uri if part else None,
                    row.audio_part_id,
                )
            else:
                status = row.status
                mapping = {
                    "separating": Disposition.ALREADY_PROCESSING,
                    "rejected": Disposition.ALREADY_REJECTED,
                }
                if status in {"separated", "transcribing", "transcribed"}:
                    raise RuntimeError("invalid_separation_state")
                if status not in mapping:
                    raise RuntimeError("invalid_chunk_state")
                return Claim(identifier, mapping[status], status)
        if contract_drift:
            raise RuntimeError("parent_contract_drift")
        return claimed

    def complete(self, claim, model, snapshot, result):
        with self.factory.begin() as session:
            row = session.get(Chunk, claim.chunk_id, with_for_update=True)
            part = (
                session.get(AudioPart, row.audio_part_id) if row is not None else None
            )
            if (
                not row
                or row.status != "separating"
                or row.audio_part_id != claim.audio_part_id
                or row.audio_uri != claim.audio_uri
                or row.duration_ms != claim.duration_ms
                or row.relative_start_ms != claim.start_ms
                or row.relative_end_ms != claim.end_ms
                or row.lang != claim.lang
                or not part
                or part.status != "completed"
                or part.duration_ms != claim.part_duration_ms
                or part.diarization_uri != claim.diarization_uri
                or part.lang != claim.lang
            ):
                raise RuntimeError("completion_conflict")
            current = dict(row.final_results or {})
            existing = current.get("separation")
            if existing is not None and existing != result:
                raise RuntimeError("separation_conflict")
            current["separation"] = result
            row.diarization_model = model
            row.diarizations = snapshot
            row.final_results = current
            row.status = "separated"
            row.error = None

    def reject(self, identifier, error):
        self._mark(identifier, "rejected", error)

    def fail(self, identifier, error):
        self._mark(identifier, "failed", error)

    def _mark(self, identifier, status, error):
        with self.factory.begin() as session:
            result = session.execute(
                update(Chunk)
                .where(Chunk.id == identifier, Chunk.status == "separating")
                .values(status=status, error=error)
            )
            if result.rowcount != 1:
                raise RuntimeError("state_transition_conflict")

    def close(self):
        if self.engine:
            self.engine.dispose()
