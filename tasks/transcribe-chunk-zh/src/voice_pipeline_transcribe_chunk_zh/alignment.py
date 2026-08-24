from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .config import UtterancePolicy

SENTENCE_END = ("。", "！", "？", "!", "?", "…")
CLAUSE_END = ("，", "、", "；", ",", ";")
PUNCTUATION = frozenset("。，、；：？！…—～·．,;:?!")


@dataclass(frozen=True, slots=True)
class DecodedUnit:
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class AlignedUnit:
    start_ms: int
    end_ms: int
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Utterance:
    start_ms: int
    end_ms: int
    text: str
    confidence: float


def _contains_cjk(value: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in value)


def normalize_units(
    decoded: list[DecodedUnit],
    *,
    offset_ms: int,
    duration_ms: int,
    policy: UtterancePolicy,
) -> list[AlignedUnit]:
    units: list[AlignedUnit] = []
    for item in decoded:
        text = item.text.strip()
        if (
            not text
            or any(char in PUNCTUATION for char in text)
            or (_contains_cjk(text) and len(text) != 1)
            or not math.isfinite(item.start_seconds)
            or not math.isfinite(item.end_seconds)
            or not math.isfinite(item.confidence)
            or not 0 <= item.confidence <= 1
        ):
            raise ValueError("decoded alignment unit is invalid")
        start = offset_ms + round(item.start_seconds * 1000)
        end = offset_ms + round(item.end_seconds * 1000)
        if end - start > policy.character_max_duration_ms:
            end = start + policy.character_capped_duration_ms
        if not 0 <= start < end <= duration_ms:
            raise ValueError("decoded alignment timestamp is out of bounds")
        if units and (start, end) < (units[-1].start_ms, units[-1].end_ms):
            raise ValueError("decoded alignment units are not in canonical order")
        units.append(AlignedUnit(start, end, text, item.confidence))
    return units


def attach_punctuation(
    units: list[AlignedUnit], punctuated_text: str
) -> list[AlignedUnit]:
    """Attach inserted punctuation to the preceding timed unit without rewriting text."""
    if not units:
        if punctuated_text.strip():
            raise ValueError("punctuation output exists for empty alignment")
        return []
    source: list[tuple[str, int]] = []
    for index, unit in enumerate(units):
        source.extend((char, index) for char in unit.text if not char.isspace())
    output = list(units)
    cursor = 0
    last_unit: int | None = None
    for char in punctuated_text:
        if char.isspace():
            continue
        if cursor < len(source) and char.casefold() == source[cursor][0].casefold():
            last_unit = source[cursor][1]
            cursor += 1
            continue
        if char in PUNCTUATION and last_unit is not None:
            output[last_unit] = replace(
                output[last_unit], text=output[last_unit].text + char
            )
            continue
        raise ValueError("punctuation model rewrote or reordered transcript text")
    if cursor != len(source):
        raise ValueError("punctuation model dropped transcript text")
    return output


def restore_punctuation(
    units: list[AlignedUnit], punctuation_model, *, max_chars: int
) -> list[AlignedUnit]:
    """Run bounded punctuation inference and retain every timed source unit."""
    if max_chars <= 0:
        raise ValueError("punctuation max_chars must be positive")
    restored: list[AlignedUnit] = []
    batch: list[AlignedUnit] = []
    character_count = 0

    def flush() -> None:
        nonlocal character_count
        if not batch:
            return
        plain_text = "".join(item.text for item in batch)
        restored.extend(
            attach_punctuation(batch, punctuation_model.restore(plain_text))
        )
        batch.clear()
        character_count = 0

    for unit in units:
        unit_characters = sum(not char.isspace() for char in unit.text)
        if unit_characters > max_chars:
            raise ValueError("one alignment unit exceeds punctuation input limit")
        if batch and character_count + unit_characters > max_chars:
            flush()
        batch.append(unit)
        character_count += unit_characters
    flush()
    return restored


def build_utterances(
    units: list[AlignedUnit], policy: UtterancePolicy
) -> list[Utterance]:
    if not units:
        return []
    groups: list[list[AlignedUnit]] = [[units[0]]]
    for unit in units[1:]:
        group = groups[-1]
        previous = group[-1]
        gap = unit.start_ms - previous.end_ms
        running_duration = previous.end_ms - group[0].start_ms
        should_split = (
            previous.text.endswith(SENTENCE_END)
            or gap > policy.huge_gap_ms
            or (
                running_duration >= policy.clause_min_ms
                and previous.text.endswith(CLAUSE_END)
            )
            or (running_duration > policy.medium_min_ms and gap > policy.medium_gap_ms)
            or (
                running_duration > policy.emergency_min_ms
                and gap >= policy.emergency_gap_ms
            )
        )
        if should_split:
            groups.append([unit])
        else:
            group.append(unit)
    return [
        Utterance(
            group[0].start_ms,
            group[-1].end_ms,
            "".join(item.text for item in group),
            sum(item.confidence for item in group) / len(group),
        )
        for group in groups
    ]
