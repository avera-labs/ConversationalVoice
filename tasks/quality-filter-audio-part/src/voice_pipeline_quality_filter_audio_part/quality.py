"""Per-speech quality decisions and good-region construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from voice_pipeline_diarization_artifact import DiarizationTurn

from .config import QualityPolicy
from .intervals import Interval, overlap_ms, total_overlap_ms


@dataclass(frozen=True, slots=True)
class SpeechQuality:
    interval: Interval
    snr_db: float
    music_overlap_ratio: float
    is_good: bool


@dataclass(frozen=True, slots=True)
class QualityGroup:
    is_good: bool
    intervals: tuple[Interval, ...]

    @property
    def span(self) -> Interval:
        return Interval(self.intervals[0].start_ms, self.intervals[-1].end_ms)


def decide_quality(
    intervals: tuple[Interval, ...],
    snr_values: tuple[float, ...],
    music: tuple[Interval, ...],
    policy: QualityPolicy,
) -> tuple[SpeechQuality, ...]:
    if len(intervals) != len(snr_values):
        raise ValueError("SNR results do not match speech intervals")
    decisions = []
    for interval, snr_db in zip(intervals, snr_values, strict=True):
        if not math.isfinite(snr_db):
            raise ValueError("SNR result is not finite")
        ratio = total_overlap_ms(interval, music) / interval.duration_ms
        is_good = snr_db >= policy.min_snr_db and ratio <= policy.max_music_overlap_ratio
        decisions.append(SpeechQuality(interval, snr_db, ratio, is_good))
    return tuple(decisions)


def _group(decisions: tuple[SpeechQuality, ...]) -> list[QualityGroup]:
    groups: list[QualityGroup] = []
    for decision in decisions:
        if groups and groups[-1].is_good == decision.is_good:
            previous = groups[-1]
            groups[-1] = QualityGroup(previous.is_good, previous.intervals + (decision.interval,))
        else:
            groups.append(QualityGroup(decision.is_good, (decision.interval,)))
    return groups


def _absorb_short_bad(groups: list[QualityGroup], threshold_ms: int) -> list[QualityGroup]:
    changed = True
    while changed:
        changed = False
        output: list[QualityGroup] = []
        index = 0
        while index < len(groups):
            if (
                0 < index < len(groups) - 1
                and not groups[index].is_good
                and groups[index].span.duration_ms < threshold_ms
                and output
                and output[-1].is_good
                and groups[index + 1].is_good
            ):
                left = output.pop()
                output.append(
                    QualityGroup(True, left.intervals + groups[index].intervals + groups[index + 1].intervals)
                )
                index += 2
                changed = True
            else:
                output.append(groups[index])
                index += 1
        groups = output
    return groups


def build_good_regions(
    decisions: tuple[SpeechQuality, ...],
    music: tuple[Interval, ...],
    policy: QualityPolicy,
) -> tuple[Interval, ...]:
    groups = _absorb_short_bad(_group(decisions), policy.max_absorbable_bad_group_ms)
    regions: list[Interval] = []
    for group in groups:
        if not group.is_good:
            continue
        current: list[Interval] = [group.intervals[0]]
        for previous, following in zip(group.intervals, group.intervals[1:]):
            gap_has_music = previous.end_ms < following.start_ms and any(
                overlap_ms(Interval(previous.end_ms, following.start_ms), item) > 0
                for item in music
            )
            if gap_has_music:
                span = Interval(current[0].start_ms, current[-1].end_ms)
                if span.duration_ms >= policy.min_good_region_ms:
                    regions.append(span)
                current = [following]
            else:
                current.append(following)
        span = Interval(current[0].start_ms, current[-1].end_ms)
        if span.duration_ms >= policy.min_good_region_ms:
            regions.append(span)
    return tuple(regions)


def align_regions_to_turns(
    regions: tuple[Interval, ...], turns: tuple[DiarizationTurn, ...]
) -> tuple[Interval, ...]:
    aligned: list[Interval] = []
    for region in regions:
        overlapping = tuple(
            turn
            for turn in turns
            if turn.end_ms > region.start_ms and turn.start_ms < region.end_ms
        )
        if not overlapping:
            continue
        start_ms = min(turn.start_ms for turn in overlapping)
        end_ms = max(turn.end_ms for turn in overlapping)
        aligned.append(Interval(start_ms, end_ms))
    return tuple(sorted(aligned))
