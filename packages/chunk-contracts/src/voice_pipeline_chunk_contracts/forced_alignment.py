"""Canonical word and zero-duration audio-tag alignment helpers."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contract import ChunkContractError
from .tagged_text import parse_text_with_audio_tags

_ITEM_FIELDS = {
    "item_index",
    "type",
    "text",
    "text_start",
    "text_end",
    "start_ms",
    "end_ms",
}


@dataclass(frozen=True, slots=True)
class AlignedTextUnit:
    """One segment-relative unit returned by the forced aligner."""

    text: str
    start_ms: int
    end_ms: int


def build_segment_word_alignment(
    text_with_audio_tags: str,
    aligned_units: Sequence[AlignedTextUnit],
    *,
    duration_ms: int,
) -> list[dict[str, object]]:
    """Merge model word spans with inline tags on one TTS segment timeline."""

    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ValueError("alignment duration must be a positive integer")
    tagged = parse_text_with_audio_tags(text_with_audio_tags)
    words = _word_items(tagged.text, aligned_units, duration_ms)
    tags = [
        {
            "type": "audio_tag",
            "text": tag,
            "text_start": offset,
            "text_end": offset,
            "start_ms": _tag_anchor(offset, words),
            "end_ms": _tag_anchor(offset, words),
            "_order": order,
        }
        for order, (tag, offset) in enumerate(
            zip(tagged.tags, tagged.tag_offsets, strict=True)
        )
    ]
    combined = [*words, *tags]
    combined.sort(
        key=lambda item: (
            item["text_start"],
            0 if item["type"] == "audio_tag" else 1,
            item.get("_order", 0),
        )
    )
    return [
        {
            "item_index": index,
            **{key: value for key, value in item.items() if key != "_order"},
        }
        for index, item in enumerate(combined)
    ]


def offset_word_alignment(
    alignment: Sequence[Mapping[str, object]], offset_ms: int
) -> list[dict[str, object]]:
    """Move one segment-relative alignment onto its assembled track timeline."""

    if isinstance(offset_ms, bool) or not isinstance(offset_ms, int) or offset_ms < 0:
        raise ValueError("alignment offset must be a non-negative integer")
    return [
        {
            **item,
            "start_ms": item["start_ms"] + offset_ms,
            "end_ms": item["end_ms"] + offset_ms,
        }
        for item in alignment
    ]


def fit_segment_word_alignment(
    text_with_audio_tags: str,
    alignment: Sequence[Mapping[str, object]],
    *,
    duration_ms: int,
) -> list[dict[str, object]]:
    """Fit rounded segment spans to the exact duration used by track assembly."""

    units = [
        AlignedTextUnit(
            str(item["text"]),
            min(int(item["start_ms"]), duration_ms),
            min(max(int(item["end_ms"]), int(item["start_ms"])), duration_ms),
        )
        for item in alignment
        if item.get("type") == "word"
    ]
    return build_segment_word_alignment(
        text_with_audio_tags,
        units,
        duration_ms=duration_ms,
    )


def validate_utterance_word_alignment(
    value: object,
    *,
    text_with_audio_tags: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, object]]:
    """Validate global word/tag spans by reconstructing their canonical merge."""

    if not isinstance(value, list):
        raise ChunkContractError("word_alignment must be an array")
    units: list[AlignedTextUnit] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _ITEM_FIELDS:
            raise ChunkContractError("word alignment item fields are invalid")
        if raw["item_index"] != index or raw["type"] not in {"word", "audio_tag"}:
            raise ChunkContractError("word alignment item identity is invalid")
        if not isinstance(raw["text"], str) or not raw["text"]:
            raise ChunkContractError("word alignment text is invalid")
        for key in ("text_start", "text_end", "start_ms", "end_ms"):
            if isinstance(raw[key], bool) or not isinstance(raw[key], int):
                raise ChunkContractError("word alignment bounds are invalid")
        if raw["type"] == "word":
            units.append(
                AlignedTextUnit(
                    raw["text"],
                    raw["start_ms"] - start_ms,
                    raw["end_ms"] - start_ms,
                )
            )
    try:
        expected = offset_word_alignment(
            build_segment_word_alignment(
                text_with_audio_tags,
                units,
                duration_ms=end_ms - start_ms,
            ),
            start_ms,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ChunkContractError("word alignment is invalid") from exc
    if expected != value:
        raise ChunkContractError("word alignment disagrees with tagged text or timing")
    return [dict(item) for item in value]


def _word_items(
    text: str, aligned_units: Sequence[AlignedTextUnit], duration_ms: int
) -> list[dict[str, object]]:
    kept = [(character, index) for index, character in enumerate(text) if _kept(character)]
    source = "".join(character for character, _index in kept)
    cursor = 0
    previous_start = 0
    words: list[dict[str, object]] = []
    for unit in aligned_units:
        cleaned = "".join(character for character in unit.text if _kept(character))
        if (
            not cleaned
            or source[cursor : cursor + len(cleaned)] != cleaned
            or isinstance(unit.start_ms, bool)
            or not isinstance(unit.start_ms, int)
            or isinstance(unit.end_ms, bool)
            or not isinstance(unit.end_ms, int)
            or not 0 <= unit.start_ms <= unit.end_ms <= duration_ms
            or unit.start_ms < previous_start
        ):
            raise ValueError("forced alignment unit is invalid")
        text_start = kept[cursor][1]
        cursor += len(cleaned)
        text_end = kept[cursor - 1][1] + 1
        words.append(
            {
                "type": "word",
                "text": cleaned,
                "text_start": text_start,
                "text_end": text_end,
                "start_ms": unit.start_ms,
                "end_ms": unit.end_ms,
            }
        )
        previous_start = unit.start_ms
    if cursor != len(kept):
        raise ValueError("forced alignment does not cover the supplied text")
    return words


def _tag_anchor(offset: int, words: Sequence[Mapping[str, object]]) -> int:
    if not words:
        return 0
    previous_end = 0
    previous_text_end: int | None = None
    for word in words:
        text_start = word["text_start"]
        text_end = word["text_end"]
        if offset <= text_start:
            if previous_text_end is not None and offset == previous_text_end:
                return previous_end
            return word["start_ms"]
        if text_start < offset < text_end:
            ratio = (offset - text_start) / (text_end - text_start)
            return word["start_ms"] + round(
                ratio * (word["end_ms"] - word["start_ms"])
            )
        previous_end = word["end_ms"]
        previous_text_end = text_end
    return previous_end


def _kept(character: str) -> bool:
    return character == "'" or unicodedata.category(character)[0] in {"L", "N"}
