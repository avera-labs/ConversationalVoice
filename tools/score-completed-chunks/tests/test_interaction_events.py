from __future__ import annotations

import numpy as np

from voice_pipeline_score_completed_chunks.audio import Audio
from voice_pipeline_score_completed_chunks.interaction_config import InteractionConfig
from voice_pipeline_score_completed_chunks.interaction_events import (
    build_stage_analysis,
    stage_metrics,
)
from voice_pipeline_score_completed_chunks.interaction_transcript import (
    InteractionUtterance,
)
from voice_pipeline_score_completed_chunks.vad import EnergyVad, Interval


def utterance(
    index: int,
    speaker: int,
    start: int,
    end: int,
    text: str,
) -> InteractionUtterance:
    return InteractionUtterance(index, speaker, start, end, text, source_index=index)


def test_energy_vad_uses_full_track_and_filters_silence() -> None:
    rate = 16_000
    samples = np.zeros(rate * 2, dtype=np.float32)
    samples[rate // 2 : rate] = 0.2
    audio = Audio(samples, rate, samples.size, 2000)
    intervals = EnergyVad(InteractionConfig()).intervals(audio)
    assert intervals == (Interval(480, 1020),)


def test_default_overlap_threshold_requires_two_vad_frames() -> None:
    config = InteractionConfig()
    assert config.minimum_cross_speaker_overlap_ms == 2 * config.vad_frame_ms == 60

    analysis = build_stage_analysis(
        utterances=(),
        activities=((Interval(0, 1000),), (Interval(940, 1000),)),
        duration_ms=1000,
        config=config,
    )
    assert analysis.overlap_intervals == (Interval(940, 1000),)
    assert analysis.overlap_event_intervals == (Interval(940, 1000),)


def test_overlap_event_count_merges_qualifying_fragments_within_500_ms() -> None:
    config = InteractionConfig()
    assert config.overlap_event_merge_gap_ms == 500

    analysis = build_stage_analysis(
        utterances=(),
        activities=(
            (
                Interval(0, 100),
                Interval(600, 700),
                Interval(1201, 1301),
            ),
            (Interval(0, 1301),),
        ),
        duration_ms=1301,
        config=config,
    )

    assert analysis.overlap_intervals == (
        Interval(0, 100),
        Interval(600, 700),
        Interval(1201, 1301),
    )
    assert analysis.overlap_event_intervals == (
        Interval(0, 700),
        Interval(1201, 1301),
    )
    assert stage_metrics(analysis)["overlap_event_count"] == 2


def test_short_overlap_fragment_cannot_bridge_two_events() -> None:
    analysis = build_stage_analysis(
        utterances=(),
        activities=(
            (Interval(0, 100), Interval(500, 550), Interval(1000, 1100)),
            (Interval(0, 1100),),
        ),
        duration_ms=1100,
        config=InteractionConfig(),
    )

    assert analysis.overlap_intervals == (Interval(0, 100), Interval(1000, 1100))
    assert analysis.overlap_event_intervals == (
        Interval(0, 100),
        Interval(1000, 1100),
    )
    assert stage_metrics(analysis)["overlap_event_count"] == 2


def test_backchannel_and_interruption_taxonomy() -> None:
    config = InteractionConfig()
    backchannel = build_stage_analysis(
        utterances=(
            utterance(0, 0, 0, 1000, "main point"),
            utterance(1, 1, 800, 950, "yeah"),
        ),
        activities=((Interval(0, 1000),), (Interval(800, 950),)),
        duration_ms=1200,
        config=config,
    )
    assert [event.category for event in backchannel.transitions] == ["backchannel"]
    assert stage_metrics(backchannel)["overlap_transition_rate"] == 1.0

    interruption = build_stage_analysis(
        utterances=(
            utterance(0, 0, 0, 1000, "main point"),
            utterance(1, 1, 800, 1500, "let me answer"),
        ),
        activities=((Interval(0, 1000),), (Interval(800, 1500),)),
        duration_ms=1600,
        config=config,
    )
    assert [event.category for event in interruption.transitions] == ["interruption"]
    assert interruption.transitions[0].overlap_duration_ms == 200


def test_transcript_attribution_does_not_clip_detected_boundaries() -> None:
    analysis = build_stage_analysis(
        utterances=(utterance(0, 0, 100, 1000, "speech"),),
        activities=((Interval(60, 1100),), ()),
        duration_ms=1200,
        config=InteractionConfig(),
    )
    acoustic = analysis.acoustic_utterances[0]
    assert acoustic.start_ms == 60
    assert acoustic.end_ms == 1100
