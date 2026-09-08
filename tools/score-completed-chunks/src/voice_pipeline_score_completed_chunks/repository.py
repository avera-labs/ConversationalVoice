from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator
from uuid import UUID

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True, slots=True)
class CompletedChunk:
    chunk_id: UUID
    language: str
    updated_at: datetime
    audio_part_audio_uri: str
    final_results: dict[str, Any]
    source_cluster_id: UUID | None = None
    chunk_audio_uri: str | None = None


def _connection_url(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


class Repository:
    def __init__(self, database_url: str):
        self.database_url = _connection_url(database_url)

    def iter_completed(
        self,
        *,
        chunk_ids: tuple[UUID, ...] = (),
        limit: int | None = None,
    ) -> Iterator[CompletedChunk]:
        conditions = ["c.status = 'completed'"]
        parameters: list[object] = []
        if chunk_ids:
            conditions.append("c.id = ANY(%s)")
            parameters.append(list(chunk_ids))
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT %s"
            parameters.append(limit)
        query = f"""
            SELECT
                c.id AS chunk_id,
                c.lang AS language,
                c.updated_at,
                c.final_results,
                c.audio_uri AS chunk_audio_uri,
                ap.audio_uri AS audio_part_audio_uri,
                ap.raw_audio_id AS source_cluster_id
            FROM chunks AS c
            JOIN audio_parts AS ap ON ap.id = c.audio_part_id
            WHERE {" AND ".join(conditions)}
            ORDER BY c.created_at, c.id
            {limit_sql}
        """
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            connection.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
            )
            with connection.cursor(name="completed_chunk_scores") as cursor:
                cursor.itersize = 50
                cursor.execute(query, parameters)
                for row in cursor:
                    results = row["final_results"]
                    if not isinstance(results, dict):
                        results = {}
                    yield CompletedChunk(
                        chunk_id=row["chunk_id"],
                        language=row["language"],
                        updated_at=row["updated_at"],
                        audio_part_audio_uri=row["audio_part_audio_uri"],
                        final_results=results,
                        source_cluster_id=row["source_cluster_id"],
                        chunk_audio_uri=row["chunk_audio_uri"],
                    )
