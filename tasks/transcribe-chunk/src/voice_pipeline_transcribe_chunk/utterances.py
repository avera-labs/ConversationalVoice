from __future__ import annotations

import math
from dataclasses import dataclass

from .config import UtterancePolicy

ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "jr.",
        "sr.",
        "prof.",
        "rev.",
        "hon.",
        "capt.",
        "sgt.",
        "lt.",
        "col.",
        "gen.",
        "cpl.",
        "maj.",
        "sen.",
        "rep.",
        "gov.",
        "pres.",
        "amb.",
        "inc.",
        "ltd.",
        "co.",
        "corp.",
        "llc.",
        "llp.",
        "plc.",
        "gmbh.",
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "cf.",
        "viz.",
        "a.m.",
        "p.m.",
        "u.s.",
        "u.k.",
        "u.n.",
        "e.u.",
        "st.",
        "ave.",
        "blvd.",
        "rd.",
        "ln.",
        "ct.",
        "no.",
        "fig.",
        "vol.",
        "pp.",
        "ch.",
        "sec.",
        "art.",
    }
)


@dataclass(frozen=True, slots=True)
class DecodedWord:
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class Word:
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


def normalize_words(
    decoded: list[DecodedWord],
    *,
    offset_ms: int,
    duration_ms: int,
    policy: UtterancePolicy,
) -> list[Word]:
    words: list[Word] = []
    for item in decoded:
        text = item.text.strip()
        if (
            not text
            or not math.isfinite(item.start_seconds)
            or not math.isfinite(item.end_seconds)
            or not math.isfinite(item.confidence)
            or not 0 <= item.confidence <= 1
        ):
            raise ValueError("decoded word is invalid")
        start = offset_ms + round(item.start_seconds * 1000)
        end = offset_ms + round(item.end_seconds * 1000)
        if end - start > policy.word_max_duration_ms:
            end = start + policy.word_capped_duration_ms
        if not 0 <= start < end <= duration_ms:
            raise ValueError("decoded word timestamp is out of bounds")
        if words and (start, end) < (words[-1].start_ms, words[-1].end_ms):
            raise ValueError("decoded words are not in canonical order")
        words.append(Word(start, end, text, item.confidence))
    return words


def _ends_sentence(text: str) -> bool:
    if text.endswith(("?", "!")):
        return True
    if not text.endswith(".") or text.lower() in ABBREVIATIONS:
        return False
    if text.count(".") >= 2:
        return False
    if len(text) == 2 and text[0].isalpha():
        return False
    return not text[:-1].isdigit()


def build_utterances(words: list[Word], policy: UtterancePolicy) -> list[Utterance]:
    if not words:
        return []
    groups: list[list[Word]] = [[words[0]]]
    for word in words[1:]:
        group = groups[-1]
        previous = group[-1]
        gap = word.start_ms - previous.end_ms
        running_duration = previous.end_ms - group[0].start_ms
        should_split = (
            _ends_sentence(previous.text)
            or gap > policy.huge_gap_ms
            or (
                running_duration >= policy.clause_min_ms and previous.text.endswith(",")
            )
            or (running_duration > policy.medium_min_ms and gap > policy.medium_gap_ms)
            or (
                running_duration > policy.emergency_min_ms
                and gap >= policy.emergency_gap_ms
            )
        )
        if should_split:
            groups.append([word])
        else:
            group.append(word)
    return [
        Utterance(
            group[0].start_ms,
            group[-1].end_ms,
            " ".join(item.text for item in group),
            sum(item.confidence for item in group) / len(group),
        )
        for group in groups
    ]
