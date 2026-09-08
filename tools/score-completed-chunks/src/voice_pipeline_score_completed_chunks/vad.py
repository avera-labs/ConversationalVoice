from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import Audio, resample
from .errors import ScoringError
from .interaction_config import InteractionConfig

TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True, order=True, slots=True)
class Interval:
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("invalid interval")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def merge_activity_intervals(
    intervals: list[Interval] | tuple[Interval, ...],
    *,
    maximum_gap_ms: int,
    minimum_duration_ms: int,
    duration_ms: int,
) -> tuple[Interval, ...]:
    if maximum_gap_ms < 0 or minimum_duration_ms <= 0 or duration_ms <= 0:
        raise ValueError("invalid interval merge configuration")
    merged: list[Interval] = []
    for interval in sorted(intervals):
        if interval.end_ms > duration_ms:
            raise ScoringError("active_interval_out_of_bounds")
        if merged and interval.start_ms - merged[-1].end_ms <= maximum_gap_ms:
            previous = merged.pop()
            merged.append(
                Interval(previous.start_ms, max(previous.end_ms, interval.end_ms))
            )
        else:
            merged.append(interval)
    return tuple(item for item in merged if item.duration_ms >= minimum_duration_ms)


def intersect_interval(left: Interval, right: Interval) -> Interval | None:
    start = max(left.start_ms, right.start_ms)
    end = min(left.end_ms, right.end_ms)
    return Interval(start, end) if end > start else None


def interval_total(intervals: tuple[Interval, ...] | list[Interval]) -> int:
    return sum(interval.duration_ms for interval in intervals)


class EnergyVad:
    """Deterministic full-track VAD used until a calibrated checkpoint is frozen.

    It deliberately does not inspect transcript boundaries. This keeps timing
    evidence independent of planned utterance start/end values.
    """

    def __init__(self, config: InteractionConfig) -> None:
        self.config = config

    def intervals(self, audio: Audio) -> tuple[Interval, ...]:
        samples = resample(audio.samples, audio.sample_rate_hz, TARGET_SAMPLE_RATE)
        if samples.size == 0:
            raise ScoringError("interaction_empty_audio")
        frame_samples = round(self.config.vad_frame_ms * TARGET_SAMPLE_RATE / 1000)
        frame_count = (samples.size + frame_samples - 1) // frame_samples
        padded = np.pad(samples, (0, frame_count * frame_samples - samples.size))
        frames = padded.reshape(frame_count, frame_samples)
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        dbfs = 20.0 * np.log10(np.maximum(rms, 1e-8))
        noise_floor = float(np.percentile(dbfs, 20))
        dynamic_range = float(np.percentile(dbfs, 80) - noise_floor)
        proposed = (
            noise_floor - 3.0
            if dynamic_range < 6.0
            else noise_floor + self.config.vad_noise_margin_db
        )
        threshold = min(
            self.config.vad_maximum_dbfs,
            max(self.config.vad_minimum_dbfs, proposed),
        )
        active = dbfs >= threshold
        raw: list[Interval] = []
        start_frame: int | None = None
        for index, is_active in enumerate((*active.tolist(), False)):
            if is_active and start_frame is None:
                start_frame = index
            elif not is_active and start_frame is not None:
                start_ms = start_frame * self.config.vad_frame_ms
                end_ms = min(audio.duration_ms, index * self.config.vad_frame_ms)
                if end_ms > start_ms:
                    raw.append(Interval(start_ms, end_ms))
                start_frame = None
        return merge_activity_intervals(
            raw,
            maximum_gap_ms=self.config.merge_inactive_gap_ms,
            minimum_duration_ms=self.config.minimum_active_segment_ms,
            duration_ms=audio.duration_ms,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "implementation": "deterministic-adaptive-frame-rms-v1",
            "sample_rate_hz": TARGET_SAMPLE_RATE,
            "frame_ms": self.config.vad_frame_ms,
            "minimum_active_segment_ms": self.config.minimum_active_segment_ms,
            "merge_inactive_gap_ms": self.config.merge_inactive_gap_ms,
            "minimum_dbfs": self.config.vad_minimum_dbfs,
            "maximum_dbfs": self.config.vad_maximum_dbfs,
            "noise_margin_db": self.config.vad_noise_margin_db,
        }
