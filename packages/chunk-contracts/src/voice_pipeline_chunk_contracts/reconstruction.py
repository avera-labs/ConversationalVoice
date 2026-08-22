"""Strict source-faithful reconstruction transcript contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .contract import ChunkContractError
from .extension import AUDIO_TAGS

_ROOT_FIELDS = {
    "schema_version",
    "language",
    "timebase",
    "source_duration_ms",
    "duration_ms",
    "speaker_mapping",
    "utterances",
}
_MAPPING_FIELDS = {"speaker_id", "diarization_speaker_id"}
_UTTERANCE_FIELDS = {
    "utterance_index",
    "speaker_id",
    "diarization_speaker_id",
    "speaker_utterance_index",
    "text",
    "confidence",
    "audio_tags",
    "tone",
    "source_start_ms",
    "source_end_ms",
    "start_ms",
    "end_ms",
    "relation",
    "anchor_utterance_index",
}
_RELATIONS = {"leading", "gap", "overlap", "simultaneous"}


def _mapping(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ChunkContractError(f"{name} fields are invalid")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ChunkContractError(f"{name} must be an integer")
    return value


def parse_reconstruction_transcript(
    value: object,
    *,
    speaker_mapping: Sequence[int],
    source_duration_ms: int,
) -> dict[str, Any]:
    """Validate reconstructed timings while preserving source utterance identity."""

    root = _mapping(value, _ROOT_FIELDS, "reconstruction transcript")
    duration_ms = _integer(root["duration_ms"], "duration_ms", 1)
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["language"] != "en"
        or root["timebase"] != "reconstruction"
        or _integer(root["source_duration_ms"], "source_duration_ms", 1)
        != source_duration_ms
    ):
        raise ChunkContractError("reconstruction transcript identity is invalid")
    if len(speaker_mapping) != 2 or len(set(speaker_mapping)) != 2:
        raise ChunkContractError("speaker mapping is invalid")
    raw_mapping = root["speaker_mapping"]
    if not isinstance(raw_mapping, list) or len(raw_mapping) != 2:
        raise ChunkContractError("speaker_mapping must contain two entries")
    for speaker_id, (raw, diarization_id) in enumerate(
        zip(raw_mapping, speaker_mapping, strict=True)
    ):
        item = _mapping(raw, _MAPPING_FIELDS, "speaker mapping entry")
        if item != {
            "speaker_id": speaker_id,
            "diarization_speaker_id": diarization_id,
        }:
            raise ChunkContractError("speaker mapping entry is invalid")
    utterances = root["utterances"]
    if not isinstance(utterances, list) or not utterances:
        raise ChunkContractError("reconstruction utterances must not be empty")
    previous_start = -1
    maximum_end = 0
    speaker_ends = [0, 0]
    for index, raw in enumerate(utterances):
        item = _mapping(raw, _UTTERANCE_FIELDS, "reconstruction utterance")
        speaker_id = _integer(item["speaker_id"], "speaker_id")
        source_start = _integer(item["source_start_ms"], "source_start_ms")
        source_end = _integer(item["source_end_ms"], "source_end_ms", 1)
        start = _integer(item["start_ms"], "start_ms")
        end = _integer(item["end_ms"], "end_ms", 1)
        confidence = item["confidence"]
        tags = item["audio_tags"]
        anchor = item["anchor_utterance_index"]
        if (
            _integer(item["utterance_index"], "utterance_index") != index
            or speaker_id not in {0, 1}
            or item["diarization_speaker_id"] != speaker_mapping[speaker_id]
            or _integer(item["speaker_utterance_index"], "speaker index") < 0
            or not isinstance(item["text"], str)
            or not item["text"].strip()
            or item["text"] != item["text"].strip()
            or isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(tags, list)
            or len(tags) > 3
            or len(set(tags)) != len(tags)
            or any(tag not in AUDIO_TAGS for tag in tags)
            or not isinstance(item["tone"], str)
            or item["tone"] != item["tone"].strip()
            or not 0 <= source_start < source_end <= source_duration_ms
            or not previous_start <= start < end <= duration_ms
            or start < speaker_ends[speaker_id]
            or item["relation"] not in _RELATIONS
            or (index == 0 and (item["relation"] != "leading" or anchor is not None))
            or (
                index > 0
                and (
                    isinstance(anchor, bool)
                    or not isinstance(anchor, int)
                    or not 0 <= anchor < index
                )
            )
        ):
            raise ChunkContractError("reconstruction utterance is invalid")
        previous_start = start
        speaker_ends[speaker_id] = end
        maximum_end = max(maximum_end, end)
    if maximum_end != duration_ms:
        raise ChunkContractError("reconstruction duration is invalid")
    return dict(root)
