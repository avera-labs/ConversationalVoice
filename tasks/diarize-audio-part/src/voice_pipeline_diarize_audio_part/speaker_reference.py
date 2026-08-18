"""Deterministic pure-speech planning and speaker-reference manifest writing."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from voice_pipeline_diarization_artifact import Segment

from .config import SpeakerReferencePolicy

SCHEMA_VERSION = 1


class SpeakerReferenceError(ValueError):
    """Raised when reference planning or serialization violates its contract."""


@dataclass(frozen=True, slots=True, order=True)
class ReferenceSegment:
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, int]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class SpeakerReferencePlan:
    speaker_id: int
    segments: tuple[ReferenceSegment, ...]

    @property
    def effective_duration_ms(self) -> int:
        return sum(segment.duration_ms for segment in self.segments)


@dataclass(frozen=True, slots=True)
class ReferenceAudio:
    uri: str
    sample_rate_hz: int
    size_bytes: int
    sha256: str
    segments: tuple[ReferenceSegment, ...]
    effective_duration_ms: int
    total_duration_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "uri": self.uri,
            "sample_rate_hz": self.sample_rate_hz,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "segments": [segment.to_dict() for segment in self.segments],
            "effective_duration_ms": self.effective_duration_ms,
            "total_duration_ms": self.total_duration_ms,
        }


@dataclass(frozen=True, slots=True)
class ManifestSpeaker:
    speaker_id: int
    reference_audio: ReferenceAudio

    def to_dict(self) -> dict[str, object]:
        return {
            "speaker_id": self.speaker_id,
            "reference_audio": self.reference_audio.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SpeakerReferenceManifest:
    speakers: tuple[ManifestSpeaker, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        speaker_ids = [speaker.speaker_id for speaker in self.speakers]
        if self.schema_version != SCHEMA_VERSION or speaker_ids != sorted(
            set(speaker_ids)
        ):
            raise SpeakerReferenceError("speaker reference manifest is not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "speakers": [speaker.to_dict() for speaker in self.speakers],
        }

    def to_json_bytes(self) -> bytes:
        return (json.dumps(self.to_dict(), ensure_ascii=True, indent=2) + "\n").encode(
            "utf-8"
        )

    def write(self, path: Path) -> int:
        payload = self.to_json_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return len(payload)


@dataclass(frozen=True, slots=True)
class _PureInterval:
    speaker_id: int
    start_ms: int
    end_ms: int


def _milliseconds(value: float, *, field: str) -> int:
    milliseconds = Decimal(str(value)) * Decimal(1000)
    if milliseconds != milliseconds.to_integral_value():
        raise SpeakerReferenceError(f"{field} must have millisecond precision")
    return int(milliseconds)


def _normalized_segments(
    segments: Iterable[Segment],
) -> tuple[tuple[int, int, int], ...]:
    normalized: list[tuple[int, int, int]] = []
    for segment in segments:
        if (
            isinstance(segment.speaker, bool)
            or not isinstance(segment.speaker, int)
            or segment.speaker < 0
        ):
            raise SpeakerReferenceError("speaker identifier is invalid")
        start_ms = _milliseconds(segment.start, field="segment start")
        end_ms = _milliseconds(segment.end, field="segment end")
        if start_ms < 0 or end_ms <= start_ms:
            raise SpeakerReferenceError("segment bounds are invalid")
        normalized.append((segment.speaker, start_ms, end_ms))
    return tuple(normalized)


def pure_intervals(segments: Iterable[Segment]) -> tuple[_PureInterval, ...]:
    """Return maximal ranges during which exactly one speaker is active."""

    events: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for speaker_id, start_ms, end_ms in _normalized_segments(segments):
        events[start_ms][speaker_id] += 1
        events[end_ms][speaker_id] -= 1

    active: dict[int, int] = {}
    previous_time: int | None = None
    result: list[_PureInterval] = []
    for timestamp in sorted(events):
        if previous_time is not None and timestamp > previous_time:
            live = [speaker_id for speaker_id, count in active.items() if count > 0]
            if len(live) == 1:
                speaker_id = live[0]
                if (
                    result
                    and result[-1].speaker_id == speaker_id
                    and result[-1].end_ms == previous_time
                ):
                    previous = result[-1]
                    result[-1] = _PureInterval(speaker_id, previous.start_ms, timestamp)
                else:
                    result.append(_PureInterval(speaker_id, previous_time, timestamp))

        for speaker_id, delta in events[timestamp].items():
            count = active.get(speaker_id, 0) + delta
            if count < 0:
                raise SpeakerReferenceError("segment events are inconsistent")
            if count == 0:
                active.pop(speaker_id, None)
            else:
                active[speaker_id] = count
        previous_time = timestamp

    if active:
        raise SpeakerReferenceError("segment events are incomplete")
    return tuple(result)


def plan_speaker_references(
    segments: Iterable[Segment], policy: SpeakerReferencePolicy
) -> tuple[SpeakerReferencePlan, ...]:
    """Select longest safe pure-speech ranges for every qualifying speaker."""

    candidates: dict[int, list[ReferenceSegment]] = defaultdict(list)
    for interval in pure_intervals(segments):
        selected = ReferenceSegment(
            interval.start_ms + policy.edge_trim_ms,
            interval.end_ms - policy.edge_trim_ms,
        )
        if selected.duration_ms > policy.min_segment_ms:
            candidates[interval.speaker_id].append(selected)

    plans: list[SpeakerReferencePlan] = []
    for speaker_id in sorted(candidates):
        ordered = sorted(
            candidates[speaker_id],
            key=lambda segment: (
                -segment.duration_ms,
                segment.start_ms,
                segment.end_ms,
            ),
        )
        remaining_ms = policy.max_speaker_effective_ms
        selected_segments: list[ReferenceSegment] = []
        for segment in ordered:
            if remaining_ms <= 0:
                break
            duration_ms = min(segment.duration_ms, remaining_ms)
            selected_segments.append(
                ReferenceSegment(segment.start_ms, segment.start_ms + duration_ms)
            )
            remaining_ms -= duration_ms
        plan = SpeakerReferencePlan(speaker_id, tuple(selected_segments))
        if plan.effective_duration_ms >= policy.min_speaker_effective_ms:
            plans.append(plan)
    return tuple(plans)


__all__ = [
    "ManifestSpeaker",
    "ReferenceAudio",
    "ReferenceSegment",
    "SpeakerReferenceError",
    "SpeakerReferenceManifest",
    "SpeakerReferencePlan",
    "plan_speaker_references",
    "pure_intervals",
]
