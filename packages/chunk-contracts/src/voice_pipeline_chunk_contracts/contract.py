"""Strict chunk-relative diarization and separation result contracts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from voice_pipeline_diarization_artifact import DiarizationTurn

_SNAPSHOT_FIELDS = {"schema_version", "timebase", "segments"}
_SEGMENT_FIELDS = {"speaker", "start_ms", "end_ms", "duration_ms"}
_SEPARATION_FIELDS = {
    "schema_version",
    "backend",
    "model",
    "input_audio",
    "speaker_audio",
    "audit",
}
_MODEL_FIELDS = {"repo_id", "revision", "config_version", "inference_steps"}
_INPUT_FIELDS = {"sample_rate_hz", "duration_ms", "size_bytes", "sha256"}
_SPEAKER_AUDIO_FIELDS = {
    "output_slot",
    "diarization_speaker_id",
    "uri",
    "sample_rate_hz",
    "duration_ms",
    "size_bytes",
    "sha256",
}
_AUDIT_FIELDS = {"verdict", "reference_speaker_id", "consistent_relation"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class ChunkContractError(ValueError):
    """Raised when a chunk contract is malformed or internally inconsistent."""


def _mapping(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ChunkContractError(f"{name} fields are invalid")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ChunkContractError(f"{name} must be an integer")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ChunkContractError(f"{name} must be a non-empty canonical string")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ChunkSegment:
    start_ms: int
    end_ms: int
    speaker: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, int]:
        return {
            "speaker": self.speaker,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class ChunkDiarization:
    segments: tuple[ChunkSegment, ...]

    @property
    def speaker_ids(self) -> tuple[int, int]:
        values = tuple(sorted({segment.speaker for segment in self.segments}))
        if len(values) != 2:
            raise ChunkContractError(
                "chunk diarization must contain exactly two speakers"
            )
        return values

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "timebase": "chunk",
            "segments": [segment.to_dict() for segment in self.segments],
        }


def build_chunk_diarization(
    turns: Iterable[DiarizationTurn], *, start_ms: int, end_ms: int
) -> ChunkDiarization:
    if start_ms < 0 or end_ms <= start_ms:
        raise ChunkContractError("chunk bounds are invalid")
    segments = []
    for turn in turns:
        clipped_start = max(start_ms, turn.start_ms)
        clipped_end = min(end_ms, turn.end_ms)
        if clipped_end > clipped_start:
            segments.append(
                ChunkSegment(
                    clipped_start - start_ms, clipped_end - start_ms, turn.speaker
                )
            )
    result = ChunkDiarization(tuple(sorted(segments)))
    _ = result.speaker_ids
    return result


def parse_chunk_diarization(value: object, *, duration_ms: int) -> ChunkDiarization:
    root = _mapping(value, _SNAPSHOT_FIELDS, "chunk diarization")
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["timebase"] != "chunk"
    ):
        raise ChunkContractError("chunk diarization version or timebase is invalid")
    raw_segments = root["segments"]
    if not isinstance(raw_segments, list):
        raise ChunkContractError("segments must be an array")
    segments = []
    previous = None
    for raw in raw_segments:
        item = _mapping(raw, _SEGMENT_FIELDS, "segment")
        speaker = _integer(item["speaker"], "speaker")
        start = _integer(item["start_ms"], "start_ms")
        end = _integer(item["end_ms"], "end_ms")
        stored_duration = _integer(item["duration_ms"], "duration_ms", 1)
        ordering = (start, end, speaker)
        if (
            end <= start
            or end > duration_ms
            or stored_duration != end - start
            or (previous and ordering < previous)
        ):
            raise ChunkContractError("segment bounds, duration, or order is invalid")
        previous = ordering
        segments.append(ChunkSegment(start, end, speaker))
    result = ChunkDiarization(tuple(segments))
    _ = result.speaker_ids
    return result


@dataclass(frozen=True, slots=True)
class SpeakerAudio:
    output_slot: int
    diarization_speaker_id: int
    uri: str
    sample_rate_hz: int
    duration_ms: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SeparationResult:
    document: Mapping[str, Any]
    speaker_audio: tuple[SpeakerAudio, SpeakerAudio]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.document)


def parse_separation_result(
    value: object,
    *,
    duration_ms: int,
    speaker_ids: tuple[int, int],
    input_size_bytes: int,
    input_sha256: str,
    output_uris: tuple[str, str],
) -> SeparationResult:
    root = _mapping(value, _SEPARATION_FIELDS, "separation")
    if (
        _integer(root["schema_version"], "schema_version") != 1
        or root["backend"] != "dialogue_sidon"
    ):
        raise ChunkContractError("separation version or backend is invalid")
    model = _mapping(root["model"], _MODEL_FIELDS, "model")
    if (
        model["repo_id"] != "sarulab-speech/DialogueSidon"
        or model["config_version"] != "sidon-v1"
    ):
        raise ChunkContractError("model identity is invalid")
    if not _REVISION.fullmatch(_string(model["revision"], "revision")):
        raise ChunkContractError("model revision is invalid")
    _integer(model["inference_steps"], "inference_steps", 1)
    input_audio = _mapping(root["input_audio"], _INPUT_FIELDS, "input_audio")
    if (
        _integer(input_audio["sample_rate_hz"], "sample_rate_hz") != 16000
        or _integer(input_audio["duration_ms"], "duration_ms", 1) != duration_ms
        or _integer(input_audio["size_bytes"], "size_bytes", 1) != input_size_bytes
        or input_audio["sha256"] != input_sha256
        or not _SHA256.fullmatch(_string(input_audio["sha256"], "sha256"))
    ):
        raise ChunkContractError("input audio identity is invalid")
    raw_speakers = root["speaker_audio"]
    if not isinstance(raw_speakers, list) or len(raw_speakers) != 2:
        raise ChunkContractError("speaker_audio must contain two entries")
    speakers = []
    for expected_slot, raw in enumerate(raw_speakers):
        item = _mapping(raw, _SPEAKER_AUDIO_FIELDS, "speaker_audio entry")
        slot = _integer(item["output_slot"], "output_slot")
        uri = _string(item["uri"], "uri")
        sha = _string(item["sha256"], "sha256")
        if (
            slot != expected_slot
            or uri != output_uris[slot]
            or _integer(item["sample_rate_hz"], "sample_rate_hz") != 16000
            or _integer(item["duration_ms"], "duration_ms", 1) != duration_ms
            or _integer(item["size_bytes"], "size_bytes", 1) <= 0
            or not _SHA256.fullmatch(sha)
        ):
            raise ChunkContractError("speaker audio identity is invalid")
        speakers.append(
            SpeakerAudio(
                slot,
                _integer(item["diarization_speaker_id"], "diarization_speaker_id"),
                uri,
                16000,
                duration_ms,
                item["size_bytes"],
                sha,
            )
        )
    normalized_ids = tuple(sorted(speaker_ids))
    if (
        tuple(sorted(item.diarization_speaker_id for item in speakers))
        != normalized_ids
    ):
        raise ChunkContractError("speaker mapping is not a bijection")
    audit = _mapping(root["audit"], _AUDIT_FIELDS, "audit")
    relation = audit["consistent_relation"]
    if (
        audit["verdict"] != "ok"
        or audit["reference_speaker_id"] != normalized_ids[0]
        or relation not in {"direct", "swapped"}
    ):
        raise ChunkContractError("audit is invalid")
    expected_mapping = (
        normalized_ids if relation == "direct" else tuple(reversed(normalized_ids))
    )
    if tuple(item.diarization_speaker_id for item in speakers) != expected_mapping:
        raise ChunkContractError("audit and speaker mapping disagree")
    return SeparationResult(dict(root), (speakers[0], speakers[1]))
