import pytest
from voice_pipeline_chunk_contracts import ChunkDiarization, ChunkSegment

from voice_pipeline_separate_chunk.errors import QualityRejection
from voice_pipeline_separate_chunk.windows import plan_windows


def test_default_window_policy_uses_source_overlap(policy):
    assert policy.window.maximum_ms == 120000
    assert policy.window.overlap_ms == 10000
    assert policy.window.crossfade_ms == 5000


def test_planner_accepts_inclusive_evidence_and_long_chunks(policy):
    snapshot = ChunkDiarization(
        (
            ChunkSegment(0, 9000, 0),
            ChunkSegment(10000, 19000, 1),
            ChunkSegment(20000, 29000, 0),
            ChunkSegment(30000, 39000, 1),
        )
    )
    windows = plan_windows(snapshot, 39000, policy.window)
    assert windows[0].start_ms == 0 and windows[-1].end_ms == 39000


def test_final_window_without_two_speakers_is_rejected(policy):
    snapshot = ChunkDiarization(
        (
            ChunkSegment(0, 9000, 0),
            ChunkSegment(10000, 19000, 1),
            ChunkSegment(20000, 140000, 0),
        )
    )
    with pytest.raises(QualityRejection):
        plan_windows(snapshot, 140000, policy.window)
