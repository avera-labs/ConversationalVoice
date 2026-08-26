"""Opaque language identifiers shared by chunk pipeline contracts."""

from __future__ import annotations

from .contract import ChunkContractError

ChunkLanguage = str


def parse_chunk_language(value: object) -> ChunkLanguage:
    """Validate storage shape without imposing a pipeline-wide support list."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ChunkContractError("chunk language must be a non-empty canonical string")
    return value
