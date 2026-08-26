"""Language identifiers shared by chunk pipeline contracts."""

from __future__ import annotations

from voice_pipeline_task_contracts import (
    is_chinese_language,
    parse_language_identifier,
    primary_language,
)

from .contract import ChunkContractError

ChunkLanguage = str


def parse_chunk_language(value: object) -> ChunkLanguage:
    """Validate language syntax without imposing a model support list."""
    try:
        return parse_language_identifier(value)
    except ValueError as exc:
        raise ChunkContractError(str(exc)) from exc


__all__ = [
    "ChunkLanguage",
    "is_chinese_language",
    "parse_chunk_language",
    "primary_language",
]
