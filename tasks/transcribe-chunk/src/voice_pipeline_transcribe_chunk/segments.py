from __future__ import annotations

from dataclasses import dataclass

from voice_pipeline_chunk_contracts import ChunkDiarization

from .config import SlicePolicy


@dataclass(frozen=True, slots=True, order=True)
class SpeechSlice:
    start_ms: int
    end_ms: int


def plan_slices(
    snapshot: ChunkDiarization,
    *,
    speaker_id: int,
    duration_ms: int,
    policy: SlicePolicy,
) -> tuple[SpeechSlice, ...]:
    intervals = sorted(
        (segment.start_ms, segment.end_ms)
        for segment in snapshot.segments
        if segment.speaker == speaker_id
    )
    if not intervals:
        return ()
    merged: list[list[int]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        previous = merged[-1]
        if start - previous[1] <= policy.merge_gap_ms:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])
    return tuple(
        SpeechSlice(
            max(0, start - policy.pad_ms), min(duration_ms, end + policy.pad_ms)
        )
        for start, end in merged
        if min(duration_ms, end + policy.pad_ms) > max(0, start - policy.pad_ms)
    )
