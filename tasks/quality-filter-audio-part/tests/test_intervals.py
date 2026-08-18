import pytest

from voice_pipeline_quality_filter_audio_part.intervals import (
    Interval,
    merge_intervals,
    overlap_ms,
    rational_to_milliseconds,
)


def test_overlap_and_union_use_half_open_milliseconds() -> None:
    assert overlap_ms(Interval(0, 1000), Interval(1000, 2000)) == 0
    assert overlap_ms(Interval(0, 1500), Interval(1000, 2000)) == 500
    assert merge_intervals(
        (Interval(1000, 2000), Interval(0, 1000), Interval(1500, 3000))
    ) == (Interval(0, 3000),)


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(1, 2000, 1), (1, 2001, 0), (3, 2000, 2)],
)
def test_rational_time_rounds_half_up(
    numerator: int, denominator: int, expected: int
) -> None:
    assert rational_to_milliseconds(numerator, denominator) == expected
