from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from voice_pipeline_chunk_contracts import ChunkSegment

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SpeakerReferenceError(ValueError):
    pass


class SpeakerReferenceUnavailable(SpeakerReferenceError):
    pass


def _integer(value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpeakerReferenceError("speaker reference integer is invalid")
    return value


def parse_reference_manifest(value: object, *, expected_speaker_ids: Sequence[int]):
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "speakers"}:
        raise SpeakerReferenceError("speaker reference manifest is invalid")
    if _integer(value["schema_version"]) != 1 or not isinstance(
        value["speakers"], list
    ):
        raise SpeakerReferenceError("speaker reference manifest is invalid")
    by_id = {}
    previous = -1
    for raw in value["speakers"]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "speaker_id",
            "reference_audio",
        }:
            raise SpeakerReferenceError("speaker reference entry is invalid")
        speaker_id = raw["speaker_id"]
        audio = raw["reference_audio"]
        speaker_id = _integer(speaker_id)
        if speaker_id <= previous:
            raise SpeakerReferenceError(
                "speaker references are not canonically ordered"
            )
        previous = speaker_id
        if speaker_id in by_id:
            raise SpeakerReferenceError("speaker reference ID is invalid")
        if not isinstance(audio, Mapping):
            raise SpeakerReferenceError("speaker reference audio is invalid")
        required = {
            "uri",
            "sample_rate_hz",
            "size_bytes",
            "sha256",
            "segments",
            "effective_duration_ms",
            "total_duration_ms",
        }
        if (
            set(audio) != required
            or not isinstance(audio["uri"], str)
            or not audio["uri"]
            or _integer(audio["sample_rate_hz"], 1) != 16000
            or _integer(audio["size_bytes"], 1) <= 0
            or not isinstance(audio["sha256"], str)
            or not _SHA256.fullmatch(audio["sha256"])
            or _integer(audio["effective_duration_ms"], 1) <= 0
            or _integer(audio["total_duration_ms"], 1) <= 0
            or not isinstance(audio["segments"], list)
            or not audio["segments"]
        ):
            raise SpeakerReferenceError("speaker reference audio is invalid")
        effective_duration_ms = 0
        for segment in audio["segments"]:
            if not isinstance(segment, Mapping) or set(segment) != {
                "start_ms",
                "end_ms",
                "duration_ms",
            }:
                raise SpeakerReferenceError("reference segment is invalid")
            start = _integer(segment["start_ms"])
            end = _integer(segment["end_ms"], 1)
            duration = _integer(segment["duration_ms"], 1)
            if end <= start or duration != end - start:
                raise SpeakerReferenceError("reference segment is invalid")
            effective_duration_ms += duration
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
    return tuple(by_id.get(item) for item in expected)


def longest_pure_interval(
    segments: Sequence[ChunkSegment],
    *,
    speaker_id: int,
    maximum_duration_ms: int,
    edge_trim_ms: int,
) -> tuple[int, int]:
    events: dict[int, dict[int, int]] = {}
    for segment in segments:
        events.setdefault(segment.start_ms, {}).setdefault(segment.speaker, 0)
        events[segment.start_ms][segment.speaker] += 1
        events.setdefault(segment.end_ms, {}).setdefault(segment.speaker, 0)
        events[segment.end_ms][segment.speaker] -= 1
    active: dict[int, int] = {}
    previous = None
    pure: list[tuple[int, int]] = []
    for timestamp in sorted(events):
        if (
            previous is not None
            and timestamp > previous
            and {key for key, count in active.items() if count > 0} == {speaker_id}
        ):
            if pure and pure[-1][1] == previous:
                pure[-1] = (pure[-1][0], timestamp)
            else:
                pure.append((previous, timestamp))
        for key, delta in events[timestamp].items():
            active[key] = active.get(key, 0) + delta
            if active[key] == 0:
                del active[key]
        previous = timestamp
    eligible = [
        (start + edge_trim_ms, end - edge_trim_ms)
        for start, end in pure
        if end - start > edge_trim_ms * 2
    ]
    if not eligible:
        raise SpeakerReferenceUnavailable("speaker_reference_unavailable")
    start, end = min(eligible, key=lambda item: (-(item[1] - item[0]), item[0]))
    return start, min(end, start + maximum_duration_ms)
