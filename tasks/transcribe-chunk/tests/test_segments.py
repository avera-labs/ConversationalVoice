from voice_pipeline_chunk_contracts import ChunkDiarization, ChunkSegment

from voice_pipeline_transcribe_chunk.config import SlicePolicy
from voice_pipeline_transcribe_chunk.segments import SpeechSlice, plan_slices


def test_snapshot_segments_merge_inclusively_and_pad():
    snapshot = ChunkDiarization(
        (
            ChunkSegment(500, 1000, 4),
            ChunkSegment(3000, 4000, 4),
            ChunkSegment(6100, 7000, 4),
            ChunkSegment(1000, 2000, 7),
        )
    )
    result = plan_slices(
        snapshot,
        speaker_id=4,
        duration_ms=8000,
        policy=SlicePolicy(merge_gap_ms=2000, pad_ms=500),
    )
    assert result == (SpeechSlice(0, 4500), SpeechSlice(5600, 7500))


def test_empty_speaker_is_a_valid_empty_plan():
    snapshot = ChunkDiarization((ChunkSegment(0, 100, 4), ChunkSegment(100, 200, 7)))
    assert (
        plan_slices(
            snapshot,
            speaker_id=8,
            duration_ms=200,
            policy=SlicePolicy(merge_gap_ms=2000, pad_ms=500),
        )
        == ()
    )
