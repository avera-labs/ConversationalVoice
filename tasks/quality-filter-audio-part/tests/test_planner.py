from voice_pipeline_diarization_artifact import DiarizationTurn
from voice_pipeline_quality_filter_audio_part.intervals import Interval
from voice_pipeline_quality_filter_audio_part.planner import (
    RawWindow,
    is_strict_two_speaker_window,
    merge_region_windows,
    plan_chunks,
    raw_windows_for_region,
)


def turns(values):
    return tuple(DiarizationTurn(start, end, speaker) for speaker, start, end in values)


def test_source_flow_regions_plan_and_merge_independently(planner_policy) -> None:
    all_turns = turns(
        [
            (0, 0, 10000), (1, 11000, 21000), (0, 22000, 24000),
            (0, 25000, 35000), (1, 36000, 46000),
            (0, 52000, 62000), (1, 63000, 73000), (2, 74000, 76000),
            (0, 77000, 87000), (1, 88000, 98000),
        ]
    )
    drafts = plan_chunks(
        all_turns,
        (Interval(0, 46000), Interval(52000, 98000)),
        planner_policy,
    )
    assert [(item.start_ms, item.end_ms) for item in drafts] == [
        (0, 46000),
        (52000, 73000),
        (76000, 98000),
    ]


def test_merge_can_create_chunk_longer_than_sixty_seconds(planner_policy) -> None:
    alternating = turns(
        [(index % 2, index * 7000, (index + 1) * 7000) for index in range(12)]
    )
    drafts = plan_chunks(alternating, (Interval(0, 84000),), planner_policy)
    assert len(drafts) == 1
    assert drafts[0].duration_ms == 84000


def test_merge_rejects_third_speaker_in_gap(planner_policy) -> None:
    region_turns = turns(
        [(0, 0, 10000), (1, 10000, 20000), (2, 22000, 23000), (0, 25000, 35000), (1, 35000, 45000)]
    )
    merged = merge_region_windows(
        (RawWindow(0, 20000), RawWindow(25000, 45000)),
        region_turns,
        planner_policy,
    )
    assert merged == (RawWindow(0, 20000), RawWindow(25000, 45000))


def test_fully_contained_turns_are_required(planner_policy) -> None:
    values = turns([(0, 0, 10000), (1, 9000, 17000), (0, 17000, 25000)])
    assert not is_strict_two_speaker_window(values, 10000, 25000, planner_policy)


def test_empty_or_single_speaker_region_has_no_raw_windows(planner_policy) -> None:
    assert raw_windows_for_region((), Interval(0, 20000), planner_policy) == ()
    assert raw_windows_for_region(
        turns([(0, 0, 20000)]), Interval(0, 20000), planner_policy
    ) == ()
