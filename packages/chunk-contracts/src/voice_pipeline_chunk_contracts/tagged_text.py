"""Canonical inline audio-tag parsing shared by generation and contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .contract import ChunkContractError


class TaggedTextError(ChunkContractError):
    """A tagged-text validation error suitable for model correction feedback."""

    def __init__(self, code: str, requirement: str):
        super().__init__(code)
        self.code = code
        self.requirement = requirement


@dataclass(frozen=True, slots=True)
class ParsedTaggedText:
    text_with_audio_tags: str
    text: str
    tags: tuple[str, ...]


def parse_text_with_audio_tags(value: str) -> ParsedTaggedText:
    """Validate inline tags and derive plain text without losing tag positions."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise TaggedTextError(
            "malformed_audio_tag",
            "text_with_audio_tags must be a non-empty canonical string.",
        )

    # Imported lazily to keep the large canonical allowlist in one module while
    # allowing extension.py to use this parser without an import cycle.
    from .extension import AUDIO_TAGS

    segments: list[str] = []
    tags: list[str] = []
    current: list[str] = []
    consecutive_tags = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == "]":
            raise TaggedTextError(
                "malformed_audio_tag",
                "Every closing bracket must belong to one approved audio tag.",
            )
        if character != "[":
            current.append(character)
            if character not in " \t":
                consecutive_tags = 0
            index += 1
            continue

        closing = value.find("]", index + 1)
        nested = value.find("[", index + 1, closing if closing >= 0 else len(value))
        if closing < 0 or nested >= 0:
            raise TaggedTextError(
                "malformed_audio_tag",
                "Audio tags must use one non-nested, balanced pair of brackets.",
            )
        tag = value[index : closing + 1]
        if tag not in AUDIO_TAGS:
            raise TaggedTextError(
                "unknown_audio_tag",
                "Every bracketed token must exactly match an approved audio tag.",
            )
        segments.append("".join(current))
        current = []
        tags.append(tag)
        consecutive_tags += 1
        if consecutive_tags > 2:
            raise TaggedTextError(
                "too_many_consecutive_audio_tags",
                "No more than two audio tags may appear consecutively at one position.",
            )
        index = closing + 1
    segments.append("".join(current))

    text = _join_text_segments(segments)
    return ParsedTaggedText(value, text, tuple(tags))


def _join_text_segments(segments: list[str]) -> str:
    """Join tag-separated text while removing only whitespace duplicated at seams."""

    result = segments[0]
    for segment in segments[1:]:
        if result and result[-1] in " \t" and segment and segment[0] in " \t":
            segment = segment.lstrip(" \t")
        result += segment
    return result.strip(" \t")
