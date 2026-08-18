import math

import pytest
from voice_pipeline_diarization_artifact import DiarizationTurn
from voice_pipeline_quality_filter_audio_part.intervals import Interval
from voice_pipeline_quality_filter_audio_part.quality import (
    align_regions_to_turns,
    build_good_regions,
    decide_quality,
)


def test_source_flow_builds_two_music_separated_good_regions(quality_policy) -> None:
    speech = tuple(
        Interval(start, end)
        for start, end in [
            (0, 10000), (11000, 21000), (22000, 24000), (25000, 35000),
            (36000, 46000), (52000, 62000), (63000, 73000), (74000, 76000),
            (77000, 87000), (88000, 98000), (100000, 112000),
        ]
    )
    snr = (18.0, 20.0, 8.5, 16.0, 21.0, 19.0, 17.0, 22.0, 20.0, 18.0, 15.0)
    music = (Interval(47000, 51000), Interval(104000, 109000))
    decisions = decide_quality(speech, snr, music, quality_policy)
    assert not decisions[2].is_good
    assert not decisions[-1].is_good
    assert build_good_regions(decisions, music, quality_policy) == (
        Interval(0, 46000),
        Interval(52000, 98000),
    )


def test_music_overlap_equality_passes(quality_policy) -> None:
    decision = decide_quality(
        (Interval(0, 10000),),
        (10.0,),
        (Interval(0, 3000),),
        quality_policy,
    )[0]
    assert decision.is_good


def test_non_finite_snr_is_rejected(quality_policy) -> None:
    with pytest.raises(ValueError, match="finite"):
        decide_quality((Interval(0, 1000),), (math.nan,), (), quality_policy)


def test_alignment_uses_first_and_last_overlapping_turn() -> None:
    turns = (DiarizationTurn(900, 2000, 0), DiarizationTurn(3000, 5100, 1))
    assert align_regions_to_turns((Interval(1000, 5000),), turns) == (
        Interval(900, 5100),
    )
