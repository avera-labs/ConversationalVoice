from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from .config import WindowingPolicy
from .wav_io import SAMPLE_RATE, frame_to_milliseconds, milliseconds_to_frames


@dataclass(frozen=True, order=True, slots=True)
class FrameSpan:
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("start_frame must not be negative")
        if self.end_frame <= self.start_frame:
            raise ValueError("end_frame must be greater than start_frame")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True, slots=True)
class IndexedWindow:
    part_index: int
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def relative_start_ms(self) -> int:
        return frame_to_milliseconds(self.start_frame)

    @property
    def relative_end_ms(self) -> int:
        return frame_to_milliseconds(self.end_frame)

    @property
    def duration_ms(self) -> int:
        return self.relative_end_ms - self.relative_start_ms


def _seconds_to_frame(value: float) -> int:
    frames = Decimal(str(value)) * SAMPLE_RATE
    return int(frames.to_integral_value(rounding=ROUND_HALF_UP))


def normalize_segments(
    segments: Iterable[tuple[float, float]],
    *,
    audio_frame_count: int,
) -> list[FrameSpan]:
    """Normalize model seconds into sorted, merged, clamped frame spans."""

    if audio_frame_count < 0:
        raise ValueError("audio_frame_count must not be negative")

    normalized: list[FrameSpan] = []
    for start_seconds, end_seconds in segments:
        if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
            continue
        if end_seconds <= start_seconds:
            continue

        start_frame = _seconds_to_frame(start_seconds)
        end_frame = _seconds_to_frame(end_seconds)
        if end_frame <= 0 or start_frame >= audio_frame_count:
            continue

        start_frame = max(0, start_frame)
        end_frame = min(audio_frame_count, end_frame)
        if end_frame <= start_frame:
            continue
        normalized.append(FrameSpan(start_frame, end_frame))

    normalized.sort()
    merged: list[FrameSpan] = []
    for span in normalized:
        if merged and span.start_frame <= merged[-1].end_frame:
            previous = merged[-1]
            merged[-1] = FrameSpan(
                previous.start_frame,
                max(previous.end_frame, span.end_frame),
            )
        else:
            merged.append(span)
    return merged


def _group_segments(
    segments: list[FrameSpan],
    *,
    gap_threshold_frames: int,
) -> list[list[FrameSpan]]:
    if not segments:
        return []

    groups: list[list[FrameSpan]] = [[segments[0]]]
    for segment in segments[1:]:
        previous = groups[-1][-1]
        if segment.start_frame - previous.end_frame > gap_threshold_frames:
            groups.append([segment])
        else:
            groups[-1].append(segment)
    return groups


def _hard_split(span: FrameSpan, *, max_frames: int) -> list[FrameSpan]:
    windows: list[FrameSpan] = []
    start = span.start_frame
    while start < span.end_frame:
        end = min(start + max_frames, span.end_frame)
        windows.append(FrameSpan(start, end))
        start = end
    return windows


def _split_group(segments: list[FrameSpan], *, max_frames: int) -> list[FrameSpan]:
    outer = FrameSpan(segments[0].start_frame, segments[-1].end_frame)
    if outer.frame_count <= max_frames:
        return [outer]

    midpoint = outer.start_frame + outer.frame_count / 2
    candidates = [
        (
            abs(((left.end_frame + right.start_frame) / 2) - midpoint),
            index,
        )
        for index, (left, right) in enumerate(
            zip(segments, segments[1:], strict=False),
            start=1,
        )
        if right.start_frame > left.end_frame
    ]
    if not candidates:
        return _hard_split(outer, max_frames=max_frames)

    _, split_index = min(candidates)
    return [
        *_split_group(segments[:split_index], max_frames=max_frames),
        *_split_group(segments[split_index:], max_frames=max_frames),
    ]


def _apply_padding(
    span: FrameSpan,
    *,
    audio_frame_count: int,
    max_frames: int,
    pad_before_frames: int,
    pad_after_frames: int,
) -> FrameSpan:
    available_padding = max_frames - span.frame_count
    desired_before = min(span.start_frame, pad_before_frames)
    desired_after = min(audio_frame_count - span.end_frame, pad_after_frames)

    before = min(desired_before, (available_padding + 1) // 2)
    after = min(desired_after, available_padding // 2)
    remaining = available_padding - before - after

    extra_before = min(desired_before - before, remaining)
    before += extra_before
    remaining -= extra_before
    after += min(desired_after - after, remaining)

    return FrameSpan(span.start_frame - before, span.end_frame + after)


def build_windows(
    segments: Iterable[FrameSpan],
    *,
    audio_frame_count: int,
    policy: WindowingPolicy,
) -> list[IndexedWindow]:
    """Group, split, pad, and index normalized speech spans deterministically."""

    if audio_frame_count < 0:
        raise ValueError("audio_frame_count must not be negative")

    ordered = sorted(segments)
    for index, segment in enumerate(ordered):
        if segment.end_frame > audio_frame_count:
            raise ValueError("segment exceeds audio_frame_count")
        if index and segment.start_frame <= ordered[index - 1].end_frame:
            raise ValueError("segments must be normalized and non-overlapping")

    gap_threshold_frames = milliseconds_to_frames(policy.gap_threshold_ms)
    min_frames = milliseconds_to_frames(policy.min_window_ms)
    max_frames = milliseconds_to_frames(policy.max_window_ms)
    pad_before_frames = milliseconds_to_frames(policy.pad_before_ms)
    pad_after_frames = milliseconds_to_frames(policy.pad_after_ms)

    windows: list[FrameSpan] = []
    for group in _group_segments(
        ordered,
        gap_threshold_frames=gap_threshold_frames,
    ):
        outer = FrameSpan(group[0].start_frame, group[-1].end_frame)
        if outer.frame_count < min_frames:
            continue
        windows.extend(_split_group(group, max_frames=max_frames))

    padded = [
        _apply_padding(
            window,
            audio_frame_count=audio_frame_count,
            max_frames=max_frames,
            pad_before_frames=pad_before_frames,
            pad_after_frames=pad_after_frames,
        )
        for window in windows
    ]
    padded.sort()
    return [
        IndexedWindow(
            part_index=index,
            start_frame=window.start_frame,
            end_frame=window.end_frame,
        )
        for index, window in enumerate(padded)
    ]
