"""Strict chunk transcription artifact and result contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .contract import ChunkContractError, SpeakerAudio
from .language import ChunkLanguage, parse_chunk_language

_ROOT_FIELDS = {
    "schema_version",
    "backend",
    "model",
    "language",
    "timebase",
    "speakers",
}
_MODEL_FIELDS = {"repo_id", "revision", "config_version"}
_SPEAKER_FIELDS = {"output_slot", "diarization_speaker_id"}
_UTTERANCE_FIELDS = {"utterance_index", "start_ms", "end_ms", "text", "confidence"}
_WORD_FIELDS = {"word_index", "start_ms", "end_ms", "text", "confidence"}
_RESULT_FIELDS = {
    "schema_version",
    "backend",
    "model",
    "language",
    "input_speaker_audio",
    "artifacts",
}
_INPUT_FIELDS = {"output_slot", "diarization_speaker_id", "uri", "size_bytes", "sha256"}
_ARTIFACTS_FIELDS = {"transcript", "word_alignment"}
_ARTIFACT_FIELDS = {"uri", "size_bytes", "sha256"}
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MODEL_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_PARAKEET_REPO = "nvidia/parakeet-tdt-0.6b-v3"
_PARAFORMER_REPO = (
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)


def _identity(language: ChunkLanguage) -> tuple[str, str, str]:
    if language == "zh":
        return "paraformer_zh", _PARAFORMER_REPO, "paraformer-zh-v1"
    return "parakeet_tdt", _PARAKEET_REPO, "parakeet-v1"


def _mapping(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ChunkContractError(f"{name} fields are invalid")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ChunkContractError(f"{name} must be an integer")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ChunkContractError(f"{name} must be a canonical string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ChunkContractError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ChunkContractError(f"{name} must be finite and between zero and one")
    return result


def _model(value: object, *, language: ChunkLanguage) -> Mapping[str, Any]:
    model = _mapping(value, _MODEL_FIELDS, "transcription model")
    _backend, repo_id, config_version = _identity(language)
    revision = _string(model["revision"], "model revision")
    if (
        model["repo_id"] != repo_id
        or model["config_version"] != config_version
        or (language != "zh" and not _REVISION.fullmatch(revision))
        or (language == "zh" and not _MODEL_REVISION.fullmatch(revision))
    ):
        raise ChunkContractError("transcription model identity is invalid")
    return model


def parse_transcription_artifact(
    value: object,
    *,
    kind: Literal["transcript", "word_alignment"],
    duration_ms: int,
    speaker_mapping: tuple[int, int],
    expected_language: ChunkLanguage = "en",
) -> dict[str, Any]:
    """Validate one canonical transcription artifact and return a plain document."""
    root = _mapping(value, _ROOT_FIELDS, kind)
    language = parse_chunk_language(expected_language)
    backend, _repo_id, _config_version = _identity(language)
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["backend"] != backend
        or root["language"] != language
        or root["timebase"] != "chunk"
    ):
        raise ChunkContractError(f"{kind} identity is invalid")
    _model(root["model"], language=language)
    raw_speakers = root["speakers"]
    if not isinstance(raw_speakers, list) or len(raw_speakers) != 2:
        raise ChunkContractError(f"{kind} speakers must contain two entries")
    collection_key = "utterances" if kind == "transcript" else "words"
    item_fields = _UTTERANCE_FIELDS if kind == "transcript" else _WORD_FIELDS
    index_key = "utterance_index" if kind == "transcript" else "word_index"
    for slot, raw_speaker in enumerate(raw_speakers):
        speaker = _mapping(
            raw_speaker, _SPEAKER_FIELDS | {collection_key}, f"{kind} speaker"
        )
        if (
            _integer(speaker["output_slot"], "output_slot") != slot
            or _integer(speaker["diarization_speaker_id"], "speaker id")
            != speaker_mapping[slot]
        ):
            raise ChunkContractError(f"{kind} speaker mapping is invalid")
        items = speaker[collection_key]
        if not isinstance(items, list):
            raise ChunkContractError(f"{collection_key} must be an array")
        previous: tuple[int, int] | None = None
        for index, raw_item in enumerate(items):
            item = _mapping(raw_item, item_fields, f"{kind} item")
            start = _integer(item["start_ms"], "start_ms")
            end = _integer(item["end_ms"], "end_ms", 1)
            if (
                _integer(item[index_key], index_key) != index
                or not 0 <= start < end <= duration_ms
                or (previous is not None and (start, end) < previous)
            ):
                raise ChunkContractError(f"{kind} item order or bounds are invalid")
            _string(item["text"], "text")
            _number(item["confidence"], "confidence")
            previous = (start, end)
    return dict(root)


def validate_artifact_pair(
    transcript: object,
    word_alignment: object,
    *,
    duration_ms: int,
    speaker_mapping: tuple[int, int],
    expected_language: ChunkLanguage = "en",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate both artifacts and their shared identity fields."""
    first = parse_transcription_artifact(
        transcript,
        kind="transcript",
        duration_ms=duration_ms,
        speaker_mapping=speaker_mapping,
        expected_language=expected_language,
    )
    second = parse_transcription_artifact(
        word_alignment,
        kind="word_alignment",
        duration_ms=duration_ms,
        speaker_mapping=speaker_mapping,
        expected_language=expected_language,
    )
    for field in ("schema_version", "backend", "model", "language", "timebase"):
        if first[field] != second[field]:
            raise ChunkContractError("transcription artifact identities disagree")
    return first, second


def parse_transcription_result(
    value: object,
    *,
    speaker_audio: Sequence[SpeakerAudio],
    artifact_uris: tuple[str, str],
    artifact_metadata: tuple[tuple[int, str], tuple[int, str]],
    expected_language: ChunkLanguage = "en",
) -> dict[str, Any]:
    """Validate the minimal durable transcription result namespace."""
    if len(speaker_audio) != 2:
        raise ChunkContractError("speaker audio must contain two entries")
    root = _mapping(value, _RESULT_FIELDS, "transcription result")
    language = parse_chunk_language(expected_language)
    backend, _repo_id, _config_version = _identity(language)
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["backend"] != backend
        or root["language"] != language
    ):
        raise ChunkContractError("transcription result identity is invalid")
    _model(root["model"], language=language)
    raw_inputs = root["input_speaker_audio"]
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 2:
        raise ChunkContractError("input_speaker_audio must contain two entries")
    for slot, (raw, expected) in enumerate(zip(raw_inputs, speaker_audio, strict=True)):
        item = _mapping(raw, _INPUT_FIELDS, "input speaker audio")
        sha = _string(item["sha256"], "input sha256")
        if (
            _integer(item["output_slot"], "output_slot") != slot
            or _integer(item["diarization_speaker_id"], "speaker id")
            != expected.diarization_speaker_id
            or item["uri"] != expected.uri
            or _integer(item["size_bytes"], "size_bytes", 1) != expected.size_bytes
            or sha != expected.sha256
            or not _SHA256.fullmatch(sha)
        ):
            raise ChunkContractError("input speaker audio identity is invalid")
    artifacts = _mapping(root["artifacts"], _ARTIFACTS_FIELDS, "artifacts")
    for index, name in enumerate(("transcript", "word_alignment")):
        item = _mapping(artifacts[name], _ARTIFACT_FIELDS, name)
        sha = _string(item["sha256"], f"{name} sha256")
        expected_size, expected_sha = artifact_metadata[index]
        if (
            item["uri"] != artifact_uris[index]
            or _integer(item["size_bytes"], "size_bytes", 1) != expected_size
            or sha != expected_sha
            or not _SHA256.fullmatch(sha)
        ):
            raise ChunkContractError(f"{name} identity is invalid")
    return dict(root)
