"""Integer millisecond half-open interval operations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


@dataclass(frozen=True, slots=True, order=True)
class Interval:
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("interval bounds are invalid")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def rational_to_milliseconds(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("rational time is invalid")
    value = Decimal(numerator) * Decimal(1000) / Decimal(denominator)
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def overlap_ms(left: Interval, right: Interval) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def total_overlap_ms(interval: Interval, others: tuple[Interval, ...]) -> int:
    return sum(overlap_ms(interval, item) for item in merge_intervals(others))


def merge_intervals(intervals: tuple[Interval, ...]) -> tuple[Interval, ...]:
    if not intervals:
        return ()
    ordered = sorted(intervals)
    merged: list[Interval] = [ordered[0]]
    for item in ordered[1:]:
        previous = merged[-1]
        if item.start_ms <= previous.end_ms:
            merged[-1] = Interval(previous.start_ms, max(previous.end_ms, item.end_ms))
        else:
            merged.append(item)
    return tuple(merged)
