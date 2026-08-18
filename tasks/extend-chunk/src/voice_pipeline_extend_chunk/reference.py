from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from voice_pipeline_chunk_contracts import ChunkSegment

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SpeakerReferenceError(ValueError):
    """Raised when the audio-part speaker-reference manifest is invalid."""


class SpeakerReferenceUnavailable(SpeakerReferenceError):
    """Raised when a required diarization speaker has no clean reference."""


def _integer(value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpeakerReferenceError("speaker reference integer is invalid")
    return value


def parse_reference_manifest(
    value: object, *, expected_speaker_ids: Sequence[int]
) -> tuple[dict | None, dict | None]:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "speakers"}:
        raise SpeakerReferenceError("speaker reference manifest fields are invalid")
    if _integer(value["schema_version"]) != 1:
        raise SpeakerReferenceError("speaker reference schema version is invalid")
    speakers = value["speakers"]
    if not isinstance(speakers, list):
        raise SpeakerReferenceError("speaker references must be an array")
    by_id: dict[int, dict] = {}
    previous = -1
    for raw in speakers:
        if not isinstance(raw, Mapping) or set(raw) != {
            "speaker_id",
            "reference_audio",
        }:
            raise SpeakerReferenceError("speaker reference entry is invalid")
        speaker_id = _integer(raw["speaker_id"])
        if speaker_id <= previous:
            raise SpeakerReferenceError(
                "speaker references are not canonically ordered"
            )
        previous = speaker_id
        audio = raw["reference_audio"]
        fields = {
            "uri",
            "sample_rate_hz",
            "size_bytes",
            "sha256",
            "segments",
            "effective_duration_ms",
            "total_duration_ms",
        }
        if not isinstance(audio, Mapping) or set(audio) != fields:
            raise SpeakerReferenceError("reference audio fields are invalid")
        if (
            not isinstance(audio["uri"], str)
            or not audio["uri"]
            or _integer(audio["sample_rate_hz"], 1) != 16000
            or _integer(audio["size_bytes"], 1) <= 0
            or not isinstance(audio["sha256"], str)
            or not _SHA256.fullmatch(audio["sha256"])
            or _integer(audio["effective_duration_ms"], 4000) < 4000
            or _integer(audio["total_duration_ms"], 1) <= 0
            or not isinstance(audio["segments"], list)
            or not audio["segments"]
        ):
            raise SpeakerReferenceError("reference audio metadata is invalid")
        effective_duration_ms = 0
        for segment in audio["segments"]:
            if not isinstance(segment, Mapping) or set(segment) != {
                "start_ms",
                "end_ms",
                "duration_ms",
            }:
                raise SpeakerReferenceError("reference segment fields are invalid")
            start_ms = _integer(segment["start_ms"])
            end_ms = _integer(segment["end_ms"], 1)
            duration_ms = _integer(segment["duration_ms"], 1)
            if end_ms <= start_ms or duration_ms != end_ms - start_ms:
                raise SpeakerReferenceError("reference segment bounds are invalid")
            effective_duration_ms += duration_ms
        total_duration_ms = effective_duration_ms + 500 * (len(audio["segments"]) - 1)
        if (
            audio["effective_duration_ms"] != effective_duration_ms
            or audio["total_duration_ms"] != total_duration_ms
        ):
            raise SpeakerReferenceError("reference duration metadata is invalid")
        by_id[speaker_id] = dict(audio)
    expected = tuple(expected_speaker_ids)
    if len(expected) != 2 or len(set(expected)) != 2:
        raise SpeakerReferenceError("expected speaker mapping is invalid")
    return by_id.get(expected[0]), by_id.get(expected[1])


def longest_pure_interval(
    segments: Sequence[ChunkSegment],
    *,
    speaker_id: int,
    maximum_duration_ms: int = 30000,
    edge_trim_ms: int = 500,
) -> tuple[int, int]:
    """Select and edge-trim the longest pure range for one speaker."""

    if (
        isinstance(speaker_id, bool)
        or not isinstance(speaker_id, int)
        or speaker_id < 0
        or isinstance(maximum_duration_ms, bool)
        or not isinstance(maximum_duration_ms, int)
        or maximum_duration_ms <= 0
        or isinstance(edge_trim_ms, bool)
        or not isinstance(edge_trim_ms, int)
        or edge_trim_ms < 0
    ):
        raise SpeakerReferenceError("fallback reference policy is invalid")
    events: dict[int, dict[int, int]] = {}
    for segment in segments:
        start_events = events.setdefault(segment.start_ms, {})
        start_events[segment.speaker] = start_events.get(segment.speaker, 0) + 1
        end_events = events.setdefault(segment.end_ms, {})
        end_events[segment.speaker] = end_events.get(segment.speaker, 0) - 1

    active: dict[int, int] = {}
    previous_time: int | None = None
    pure: list[tuple[int, int]] = []
    for timestamp in sorted(events):
        if previous_time is not None and timestamp > previous_time:
            live = {speaker for speaker, count in active.items() if count > 0}
            if live == {speaker_id}:
                if pure and pure[-1][1] == previous_time:
                    pure[-1] = (pure[-1][0], timestamp)
                else:
                    pure.append((previous_time, timestamp))
        for speaker, delta in events[timestamp].items():
            count = active.get(speaker, 0) + delta
            if count < 0:
                raise SpeakerReferenceError("fallback segment events are invalid")
            if count == 0:
                active.pop(speaker, None)
            else:
                active[speaker] = count
        previous_time = timestamp
    if active:
        raise SpeakerReferenceError("fallback segment events are incomplete")
    eligible = [
        (start_ms + edge_trim_ms, end_ms - edge_trim_ms)
        for start_ms, end_ms in pure
        if end_ms - start_ms > edge_trim_ms * 2
    ]
    if not eligible:
        raise SpeakerReferenceUnavailable("speaker_reference_unavailable")
    start_ms, end_ms = min(
        eligible, key=lambda item: (-(item[1] - item[0]), item[0], item[1])
    )
    return start_ms, min(end_ms, start_ms + maximum_duration_ms)
