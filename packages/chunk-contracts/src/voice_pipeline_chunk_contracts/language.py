"""Canonical language codes shared by chunk pipeline contracts."""

from __future__ import annotations

from typing import Literal

from .contract import ChunkContractError

ChunkLanguage = Literal["en", "zh"]
SUPPORTED_CHUNK_LANGUAGES = frozenset({"en", "zh"})


def parse_chunk_language(value: object) -> ChunkLanguage:
    if value not in SUPPORTED_CHUNK_LANGUAGES:
        raise ChunkContractError("chunk language must be 'en' or 'zh'")
    return value  # type: ignore[return-value]
