import math

import pytest

from voice_pipeline_split_raw_audio_into_parts.config import WindowingPolicy
from voice_pipeline_split_raw_audio_into_parts.wav_io import (
    SAMPLE_RATE,
    milliseconds_to_frames,
)
from voice_pipeline_split_raw_audio_into_parts.windowing import (
    FrameSpan,
    build_windows,
    normalize_segments,
)


def _frames(seconds: int) -> int:
    return seconds * SAMPLE_RATE


def _milliseconds(value: int) -> int:
    return milliseconds_to_frames(value)


def _policy(**overrides: int) -> WindowingPolicy:
    values = {
        "gap_threshold_ms": 15_000,
        "min_window_ms": 20_000,
        "max_window_ms": 900_000,
        "pad_before_ms": 250,
        "pad_after_ms": 250,
    }
    values.update(overrides)
    return WindowingPolicy(**values)


def test_normalize_segments_clamps_sorts_merges_and_discards_invalid() -> None:
    normalized = normalize_segments(
        [
            (0.5, 1.0),
            (0.0, 0.5),
            (0.75, 1.5),
            (-1.0, 0.25),
            (2.0, 2.0),
            (math.nan, 1.0),
            (3.0, math.inf),
            (10.0, 11.0),
            (-5.0, -1.0),
        ],
        audio_frame_count=_frames(2),
    )

    assert normalized == [FrameSpan(0, _milliseconds(1_500))]


def test_float_boundaries_round_to_stable_sample_frames() -> None:
    normalized = normalize_segments(
        [(0.00003125, 0.00009375)],
        audio_frame_count=10,
    )

    assert normalized == [FrameSpan(1, 2)]


def test_gap_equal_to_threshold_stays_in_group_and_greater_gap_splits() -> None:
    windows = build_windows(
        [
            FrameSpan(_frames(0), _frames(10)),
            FrameSpan(_frames(25), _frames(30)),
            FrameSpan(_milliseconds(45_001), _frames(50)),
        ],
        audio_frame_count=_frames(60),
        policy=_policy(
            min_window_ms=1_000,
            pad_before_ms=0,
            pad_after_ms=0,
        ),
    )

    assert [(window.start_frame, window.end_frame) for window in windows] == [
        (_frames(0), _frames(30)),
        (_milliseconds(45_001), _frames(50)),
    ]


def test_minimum_window_boundary_is_inclusive() -> None:
    windows = build_windows(
        [
            FrameSpan(0, _milliseconds(19_999)),
            FrameSpan(_frames(30), _frames(50)),
        ],
        audio_frame_count=_frames(60),
        policy=_policy(
            gap_threshold_ms=0,
            pad_before_ms=0,
            pad_after_ms=0,
        ),
    )

    assert [(window.start_frame, window.end_frame) for window in windows] == [
        (_frames(30), _frames(50)),
    ]


def test_oversized_group_splits_at_gap_nearest_midpoint() -> None:
    windows = build_windows(
        [
            FrameSpan(_frames(0), _frames(40)),
            FrameSpan(_frames(50), _frames(80)),
            FrameSpan(_frames(90), _frames(130)),
        ],
        audio_frame_count=_frames(140),
        policy=_policy(
            gap_threshold_ms=20_000,
            min_window_ms=1_000,
            max_window_ms=90_000,
            pad_before_ms=0,
            pad_after_ms=0,
        ),
    )

    assert [(window.start_frame, window.end_frame) for window in windows] == [
        (_frames(0), _frames(40)),
        (_frames(50), _frames(130)),
    ]


def test_single_oversized_speech_segment_is_hard_split() -> None:
    windows = build_windows(
        [FrameSpan(0, _frames(200))],
        audio_frame_count=_frames(200),
        policy=_policy(
            min_window_ms=1_000,
            max_window_ms=90_000,
            pad_before_ms=0,
            pad_after_ms=0,
        ),
    )

    assert [(window.start_frame, window.end_frame) for window in windows] == [
        (_frames(0), _frames(90)),
        (_frames(90), _frames(180)),
        (_frames(180), _frames(200)),
    ]
    assert [window.part_index for window in windows] == [0, 1, 2]


def test_padding_clamps_to_audio_and_may_overlap_adjacent_windows() -> None:
    windows = build_windows(
        [
            FrameSpan(_frames(1), _frames(10)),
            FrameSpan(_milliseconds(10_100), _frames(20)),
        ],
        audio_frame_count=_frames(21),
        policy=_policy(gap_threshold_ms=0, min_window_ms=1_000),
    )

    assert [(window.start_frame, window.end_frame) for window in windows] == [
        (_milliseconds(750), _milliseconds(10_250)),
        (_milliseconds(9_850), _milliseconds(20_250)),
    ]
    assert windows[0].end_frame > windows[1].start_frame


def test_padding_never_breaks_the_hard_maximum() -> None:
    windows = build_windows(
        [FrameSpan(_milliseconds(100), _milliseconds(1_100))],
        audio_frame_count=_milliseconds(2_000),
        policy=_policy(
            min_window_ms=1,
            max_window_ms=1_000,
            pad_before_ms=250,
            pad_after_ms=250,
        ),
    )

    assert len(windows) == 1
    assert windows[0].frame_count == _milliseconds(1_000)
    assert windows[0].frame_count <= _milliseconds(1_000)


def test_limited_padding_capacity_is_balanced_between_both_sides() -> None:
    windows = build_windows(
        [FrameSpan(_milliseconds(500), _milliseconds(1_400))],
        audio_frame_count=_milliseconds(2_000),
        policy=_policy(
            min_window_ms=1,
            max_window_ms=1_000,
            pad_before_ms=250,
            pad_after_ms=250,
        ),
    )

    assert [(window.start_frame, window.end_frame) for window in windows] == [
        (_milliseconds(450), _milliseconds(1_450)),
    ]


def test_empty_segments_produce_no_windows() -> None:
    assert (
        build_windows(
            [],
            audio_frame_count=_frames(60),
            policy=_policy(),
        )
        == []
    )


@pytest.mark.parametrize(
    "segments",
    [
        [FrameSpan(0, 20), FrameSpan(10, 30)],
        [FrameSpan(0, 20), FrameSpan(20, 30)],
    ],
)
def test_build_windows_requires_normalized_segments(
    segments: list[FrameSpan],
) -> None:
    with pytest.raises(ValueError, match="normalized and non-overlapping"):
        build_windows(
            segments,
            audio_frame_count=100,
            policy=_policy(min_window_ms=1),
        )


def test_segment_beyond_audio_boundary_is_rejected() -> None:
    with pytest.raises(ValueError, match="segment exceeds audio_frame_count"):
        build_windows(
            [FrameSpan(0, _frames(61))],
            audio_frame_count=_frames(60),
            policy=_policy(),
        )
