"""Strict dialogue-extension script and transcript contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from .contract import ChunkContractError

AUDIO_TAGS = frozenset(
    {
        "[angry]",
        "[chuckles]",
        "[clears throat]",
        "[coughs]",
        "[crying]",
        "[curious]",
        "[excited]",
        "[exhales]",
        "[inhales deeply]",
        "[laughs]",
        "[sad]",
        "[shouts]",
        "[sighs]",
        "[surprised]",
        "[thoughtful]",
        "[whispers]",
    }
)

_SCRIPT_FIELDS = {
    "schema_version",
    "backend",
    "model",
    "language",
    "target_duration_ms",
    "speaker_mapping",
    "utterances",
    "usage",
}
_MODEL_FIELDS = {"id", "config_version"}
_MAPPING_FIELDS = {"speaker_id", "diarization_speaker_id"}
_UTTERANCE_FIELDS = {
    "utterance_index",
    "speaker_id",
    "text",
    "tone",
    "type",
    "placement",
    "audio_tags",
}
_TRANSCRIPT_FIELDS = {
    "schema_version",
    "language",
    "timebase",
    "duration_ms",
    "speaker_mapping",
    "utterances",
}
_TIMED_UTTERANCE_FIELDS = _UTTERANCE_FIELDS | {"start_ms", "end_ms"}
_USAGE_FIELDS = {"model", "in_tokens", "out_tokens", "total_tokens", "cost_usd"}
_TYPES = {"dialogue", "backchannel", "paralinguistic"}
_PLACEMENTS = {"sequential", "overlap_previous"}


def _mapping(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ChunkContractError(f"{name} fields are invalid")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ChunkContractError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ChunkContractError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ChunkContractError(f"{name} must be finite and non-negative")
    return value


def _string(value: object, name: str, *, empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or (not empty and not value)
    ):
        raise ChunkContractError(f"{name} must be a canonical string")
    return value


def _speaker_mapping(value: object, expected: Sequence[int]) -> list[dict[str, int]]:
    if len(expected) != 2 or len(set(expected)) != 2:
        raise ChunkContractError("speaker mapping must contain two distinct IDs")
    if not isinstance(value, list) or len(value) != 2:
        raise ChunkContractError("speaker_mapping must contain two entries")
    result: list[dict[str, int]] = []
    for speaker_id, (raw, diarization_id) in enumerate(
        zip(value, expected, strict=True)
    ):
        item = _mapping(raw, _MAPPING_FIELDS, "speaker_mapping entry")
        if (
            _integer(item["speaker_id"], "speaker_id") != speaker_id
            or _integer(item["diarization_speaker_id"], "diarization speaker ID")
            != diarization_id
        ):
            raise ChunkContractError("speaker_mapping is invalid")
        result.append(dict(item))
    return result


def _utterance(value: object, index: int, *, timed: bool) -> dict[str, Any]:
    fields = _TIMED_UTTERANCE_FIELDS if timed else _UTTERANCE_FIELDS
    item = _mapping(value, fields, "extension utterance")
    if _integer(item["utterance_index"], "utterance_index") != index:
        raise ChunkContractError("utterance indexes are not canonical")
    speaker_id = _integer(item["speaker_id"], "speaker_id")
    if speaker_id not in {0, 1}:
        raise ChunkContractError("utterance speaker_id must be zero or one")
    utterance_type = item["type"]
    placement = item["placement"]
    if utterance_type not in _TYPES or placement not in _PLACEMENTS:
        raise ChunkContractError("utterance type or placement is invalid")
    text = _string(item["text"], "utterance text", empty=True)
    if "[" in text or "]" in text:
        raise ChunkContractError("utterance text must not contain audio tags")
    _string(item["tone"], "utterance tone")
    tags = item["audio_tags"]
    if (
        not isinstance(tags, list)
        or len(tags) > 3
        or len(set(tags)) != len(tags)
        or any(tag not in AUDIO_TAGS for tag in tags)
    ):
        raise ChunkContractError("utterance audio_tags are invalid")
    if utterance_type == "paralinguistic":
        if not tags:
            raise ChunkContractError("paralinguistic utterance requires an audio tag")
    elif not text:
        raise ChunkContractError("spoken utterance text must not be empty")
    if placement == "overlap_previous" and utterance_type == "dialogue":
        raise ChunkContractError("ordinary dialogue must not overlap")
    result = dict(item)
    if timed:
        start = _integer(item["start_ms"], "start_ms")
        end = _integer(item["end_ms"], "end_ms", 1)
        if end <= start:
            raise ChunkContractError("utterance timing is invalid")
    return result


def parse_dialogue_extension_document(
    value: object,
    *,
    speaker_mapping: Sequence[int],
    model_id: str,
    target_duration_ms: int,
    config_version: str = "dialogue-extension-v1",
    min_utterances: int = 2,
    max_utterances: int = 200,
) -> dict[str, Any]:
    """Validate the canonical LLM-produced continuation script."""

    root = _mapping(value, _SCRIPT_FIELDS, "dialogue extension document")
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["backend"] != "openrouter"
        or root["language"] != "en"
        or _integer(root["target_duration_ms"], "target_duration_ms", 1)
        != target_duration_ms
    ):
        raise ChunkContractError("dialogue extension identity is invalid")
    model = _mapping(root["model"], _MODEL_FIELDS, "dialogue extension model")
    if (
        _string(model["id"], "dialogue model ID") != model_id
        or "/" not in model["id"]
        or model["config_version"] != config_version
    ):
        raise ChunkContractError("dialogue extension model is invalid")
    _speaker_mapping(root["speaker_mapping"], speaker_mapping)
    raw_utterances = root["utterances"]
    if (
        _integer(min_utterances, "minimum utterance count", 2) < 2
        or _integer(max_utterances, "maximum utterance count", 2) < min_utterances
        or not isinstance(raw_utterances, list)
        or not min_utterances <= len(raw_utterances) <= max_utterances
    ):
        raise ChunkContractError("dialogue extension utterance count is invalid")
    utterances = [
        _utterance(raw, index, timed=False) for index, raw in enumerate(raw_utterances)
    ]
    if {item["speaker_id"] for item in utterances} != {0, 1}:
        raise ChunkContractError("dialogue extension must use both speakers")
    if utterances[0]["placement"] != "sequential":
        raise ChunkContractError("first utterance must be sequential")
    for previous, current in pairwise(utterances):
        if (
            current["placement"] == "overlap_previous"
            and current["speaker_id"] == previous["speaker_id"]
        ):
            raise ChunkContractError("a speaker must not overlap itself")
    usage = _mapping(root["usage"], _USAGE_FIELDS, "dialogue extension usage")
    if _string(usage["model"], "usage model") != model_id:
        raise ChunkContractError("dialogue extension usage model is invalid")
    in_tokens = _integer(usage["in_tokens"], "in_tokens")
    out_tokens = _integer(usage["out_tokens"], "out_tokens")
    if _integer(usage["total_tokens"], "total_tokens") not in {
        0,
        in_tokens + out_tokens,
    }:
        raise ChunkContractError("dialogue extension token counts are invalid")
    _number(usage["cost_usd"], "cost_usd")
    return dict(root)


def parse_dialogue_extension_transcript(
    value: object,
    *,
    script: Mapping[str, Any],
    speaker_mapping: Sequence[int],
) -> dict[str, Any]:
    """Validate actual TTS timings against the approved continuation script."""

    root = _mapping(value, _TRANSCRIPT_FIELDS, "dialogue extension transcript")
    duration_ms = _integer(root["duration_ms"], "duration_ms", 1)
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["language"] != "en"
        or root["timebase"] != "dialogue_extension"
    ):
        raise ChunkContractError("dialogue extension transcript identity is invalid")
    _speaker_mapping(root["speaker_mapping"], speaker_mapping)
    raw_items = root["utterances"]
    script_items = script.get("utterances")
    if (
        not isinstance(raw_items, list)
        or not isinstance(script_items, list)
        or len(raw_items) != len(script_items)
    ):
        raise ChunkContractError("dialogue extension transcript length is invalid")
    previous: dict[str, Any] | None = None
    maximum_end_ms = 0
    for index, (raw, source) in enumerate(zip(raw_items, script_items, strict=True)):
        item = _utterance(raw, index, timed=True)
        if {field: item[field] for field in _UTTERANCE_FIELDS} != source:
            raise ChunkContractError("transcript utterance disagrees with script")
        if item["end_ms"] > duration_ms:
            raise ChunkContractError("transcript utterance exceeds duration")
        if index == 0 and item["start_ms"] != 0:
            raise ChunkContractError("transcript must start at zero")
        if previous is not None:
            if item["start_ms"] < previous["start_ms"]:
                raise ChunkContractError("transcript starts are not chronological")
            if (
                item["placement"] == "sequential"
                and item["start_ms"] < previous["end_ms"]
            ):
                raise ChunkContractError("sequential utterances must not overlap")
            if (
                item["placement"] == "overlap_previous"
                and item["start_ms"] >= previous["end_ms"]
            ):
                raise ChunkContractError("overlapping utterance misses the previous turn")
        previous = item
        maximum_end_ms = max(maximum_end_ms, item["end_ms"])
    if maximum_end_ms != duration_ms:
        raise ChunkContractError("transcript duration must equal its final audio end")
    return dict(root)
