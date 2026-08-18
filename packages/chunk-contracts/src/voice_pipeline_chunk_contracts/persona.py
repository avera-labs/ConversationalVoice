"""Strict chunk persona wire, artifact, and result contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .contract import ChunkContractError

_WIRE_FIELDS = {"scene", "speakers"}
_DOCUMENT_FIELDS = {
    "scene",
    "speakers",
    "usage",
    "schema_version",
    "backend",
    "config_version",
    "language",
    "speaker_mapping",
}
_SCENE_FIELDS = {"description", "overall_tone", "emotion_intensity"}
_SPEAKER_FIELDS = {
    "name",
    "age",
    "ethnicity",
    "gender",
    "tag",
    "alpha",
    "evidence",
    "primary_emotion",
    "secondary_emotion",
    "emotion_intensity",
    "laugh",
    "cry",
    "whisper",
    "shout",
    "sigh",
    "overall_tone",
}
_DURABLE_SPEAKER_FIELDS = _SPEAKER_FIELDS | {"speaker_id"}
_USAGE_FIELDS = {"model", "in_tokens", "out_tokens", "total_tokens", "cost_usd"}
_MAPPING_FIELDS = {"output_slot", "diarization_speaker_id"}
_RESULT_FIELDS = {
    "schema_version",
    "backend",
    "model",
    "language",
    "input_audio",
    "input_transcript",
    "artifact",
}
_MODEL_FIELDS = {"id", "config_version"}
_IDENTITY_FIELDS = {"uri", "size_bytes", "sha256"}
_EMOTIONS = {"angry", "sad", "happy", "surprised", "neutral"}
_TONES = {"aggressive", "warm", "cold", "nervous", "calm", "playful"}
_LEVELS = {"low", "medium", "high"}
_NULLABLE_STRINGS = {"name", "age", "ethnicity", "gender", "evidence"}
_EVENT_FLAGS = {"laugh", "cry", "whisper", "shout", "sigh"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ChunkContractError(f"{name} must be a canonical string")
    return value


def _speaker_ids(speaker_mapping: Sequence[int]) -> tuple[str, str]:
    if len(speaker_mapping) != 2:
        raise ChunkContractError("speaker mapping must contain two entries")
    values = tuple(_integer(value, "speaker id") for value in speaker_mapping)
    if len(set(values)) != 2:
        raise ChunkContractError("speaker mapping must be a bijection")
    return tuple(sorted((str(values[0]), str(values[1]))))


def _validate_scene(value: object) -> dict[str, Any]:
    scene = _mapping(value, _SCENE_FIELDS, "scene")
    _string(scene["description"], "scene description")
    if scene["overall_tone"] not in _TONES or scene["emotion_intensity"] not in _LEVELS:
        raise ChunkContractError("scene enums are invalid")
    return dict(scene)


def _normalize_speaker(value: object, name: str) -> dict[str, Any]:
    speaker = dict(_mapping(value, _SPEAKER_FIELDS, name))
    for field in _NULLABLE_STRINGS:
        current = speaker[field]
        if isinstance(current, str) and not current.strip():
            speaker[field] = None
        elif current is not None:
            _string(current, f"{name} {field}")
    _string(speaker["tag"], f"{name} tag")
    if (
        speaker["alpha"] not in _LEVELS
        or speaker["primary_emotion"] not in _EMOTIONS
        or speaker["secondary_emotion"] not in _EMOTIONS | {None}
        or speaker["emotion_intensity"] not in _LEVELS
        or speaker["overall_tone"] not in _TONES
    ):
        raise ChunkContractError(f"{name} enums are invalid")
    if any(not isinstance(speaker[field], bool) for field in _EVENT_FLAGS):
        raise ChunkContractError(f"{name} event flags must be booleans")
    return speaker


def parse_persona_document(
    value: object,
    *,
    speaker_mapping: Sequence[int],
    model_id: str,
    config_version: str = "persona-v1",
) -> dict[str, Any]:
    """Validate the source-compatible durable persona document."""
    root = _mapping(value, _DOCUMENT_FIELDS, "persona document")
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["backend"] != "openrouter"
        or root["config_version"] != config_version
        or root["language"] != "en"
    ):
        raise ChunkContractError("persona document identity is invalid")
    _validate_scene(root["scene"])
    expected_ids = _speaker_ids(speaker_mapping)
    speakers = root["speakers"]
    if not isinstance(speakers, list) or len(speakers) != 2:
        raise ChunkContractError("persona speakers must contain two entries")
    actual_ids = []
    for raw in speakers:
        item = _mapping(raw, _DURABLE_SPEAKER_FIELDS, "persona speaker")
        normalized = _normalize_speaker(
            {field: item[field] for field in _SPEAKER_FIELDS}, "persona speaker"
        )
        if any(normalized[field] != item[field] for field in _SPEAKER_FIELDS):
            raise ChunkContractError("durable nullable strings must be canonical")
        actual_ids.append(_string(item["speaker_id"], "speaker_id"))
    if tuple(actual_ids) != expected_ids:
        raise ChunkContractError("persona speakers are not canonically ordered")
    usage = _mapping(root["usage"], _USAGE_FIELDS, "persona usage")
    if _string(usage["model"], "persona usage model") != model_id:
        raise ChunkContractError("persona usage model is invalid")
    in_tokens = _integer(usage["in_tokens"], "in_tokens")
    out_tokens = _integer(usage["out_tokens"], "out_tokens")
    if _integer(usage["total_tokens"], "total_tokens") not in {
        0,
        in_tokens + out_tokens,
    }:
        raise ChunkContractError("persona total token count is invalid")
    _number(usage["cost_usd"], "cost_usd")
    mapping = root["speaker_mapping"]
    if not isinstance(mapping, list) or len(mapping) != 2:
        raise ChunkContractError("persona speaker_mapping must contain two entries")
    for slot, raw in enumerate(mapping):
        item = _mapping(raw, _MAPPING_FIELDS, "speaker_mapping entry")
        if (
            _integer(item["output_slot"], "output_slot") != slot
            or _integer(item["diarization_speaker_id"], "speaker id")
            != speaker_mapping[slot]
        ):
            raise ChunkContractError("persona speaker_mapping is invalid")
    return dict(root)


def _validate_identity(
    value: object, expected: tuple[str, int, str], name: str
) -> None:
    item = _mapping(value, _IDENTITY_FIELDS, name)
    uri, size, sha = expected
    if (
        item["uri"] != uri
        or _integer(item["size_bytes"], f"{name} size", 1) != size
        or item["sha256"] != sha
        or not _SHA256.fullmatch(_string(item["sha256"], f"{name} sha256"))
    ):
        raise ChunkContractError(f"{name} identity is invalid")


def parse_persona_result(
    value: object,
    *,
    model_id: str,
    input_audio: tuple[str, int, str],
    input_transcript: tuple[str, int, str],
    artifact: tuple[str, int, str],
    config_version: str = "persona-v1",
) -> dict[str, Any]:
    """Validate the minimal durable final_results.persona namespace."""
    root = _mapping(value, _RESULT_FIELDS, "persona result")
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["backend"] != "openrouter"
        or root["language"] != "en"
    ):
        raise ChunkContractError("persona result identity is invalid")
    model = _mapping(root["model"], _MODEL_FIELDS, "persona model")
    stored_model_id = _string(model["id"], "persona model ID")
    if (
        "/" not in stored_model_id
        or stored_model_id != model_id
        or model["config_version"] != config_version
    ):
        raise ChunkContractError("persona model identity is invalid")
    _validate_identity(root["input_audio"], input_audio, "input audio")
    _validate_identity(root["input_transcript"], input_transcript, "input transcript")
    _validate_identity(root["artifact"], artifact, "persona artifact")
    return dict(root)
