from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from voice_pipeline_score_completed_chunks.repository import CompletedChunk


class Disposition(StrEnum):
    READY = "ready"
    NOT_COMPLETED = "not_completed"
    ALREADY_SCORED = "already_scored"


@dataclass(frozen=True, slots=True)
class Claim:
    disposition: Disposition
    chunk: CompletedChunk | None
    source_fingerprint: str | None
    evaluation: dict | None = None


def _connection_url(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def source_fingerprint(results: dict) -> str:
    source = {key: value for key, value in results.items() if key != "evaluation"}
    payload = json.dumps(
        source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Repository:
    def __init__(self, database_url: str):
        self.database_url = _connection_url(database_url)

    @classmethod
    def create(cls, environment):
        return cls(environment.database_url)

    def claim(self, identifier: UUID, *, model_fingerprint: str) -> Claim:
        query = """
            SELECT c.id, c.status, c.lang, c.updated_at, c.audio_uri,
                   c.final_results, ap.audio_uri AS audio_part_audio_uri,
                   ap.raw_audio_id AS source_cluster_id
            FROM chunks AS c
            JOIN audio_parts AS ap ON ap.id = c.audio_part_id
            WHERE c.id = %s
        """
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            row = connection.execute(query, (identifier,)).fetchone()
        if row is None:
            raise RuntimeError("chunk_not_found")
        if row["status"] != "completed":
            return Claim(Disposition.NOT_COMPLETED, None, None)
        results = row["final_results"] if isinstance(row["final_results"], dict) else {}
        fingerprint = source_fingerprint(results)
        evaluation = results.get("evaluation")
        if (
            isinstance(evaluation, dict)
            and evaluation.get("model_fingerprint") == model_fingerprint
            and evaluation.get("source_fingerprint") == fingerprint
        ):
            return Claim(Disposition.ALREADY_SCORED, None, fingerprint, evaluation)
        chunk = CompletedChunk(
            chunk_id=row["id"],
            language=row["lang"],
            updated_at=row["updated_at"],
            audio_part_audio_uri=row["audio_part_audio_uri"],
            final_results=results,
            source_cluster_id=row["source_cluster_id"],
            chunk_audio_uri=row["audio_uri"],
        )
        return Claim(Disposition.READY, chunk, fingerprint)

    def complete(
        self,
        identifier: UUID,
        *,
        source_fingerprint_value: str,
        evaluation: dict,
    ) -> dict:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            with connection.transaction():
                row = connection.execute(
                    "SELECT status, final_results FROM chunks WHERE id = %s FOR UPDATE",
                    (identifier,),
                ).fetchone()
                if row is None or row["status"] != "completed":
                    raise RuntimeError("evaluation_completion_conflict")
                results = (
                    row["final_results"]
                    if isinstance(row["final_results"], dict)
                    else {}
                )
                if source_fingerprint(results) != source_fingerprint_value:
                    raise RuntimeError("evaluation_source_changed")
                current = results.get("evaluation")
                if (
                    isinstance(current, dict)
                    and current.get("model_fingerprint")
                    == evaluation.get("model_fingerprint")
                    and current.get("source_fingerprint") == source_fingerprint_value
                ):
                    return current
                updated = dict(results)
                updated["evaluation"] = evaluation
                connection.execute(
                    "UPDATE chunks SET final_results = %s WHERE id = %s",
                    (Jsonb(updated), identifier),
                )
        return evaluation

    def close(self):
        return None
