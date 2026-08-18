import numpy as np
import pytest
from voice_pipeline_chunk_contracts import ChunkDiarization, ChunkSegment

from voice_pipeline_separate_chunk.audit import audit_tracks
from voice_pipeline_separate_chunk.errors import QualityRejection


def test_audit_persists_direct_and_swapped_mapping(policy):
    snapshot = ChunkDiarization(
        (ChunkSegment(0, 9000, 4), ChunkSegment(10000, 19000, 7))
    )
    tracks = np.zeros((2, 19000 * 16), dtype=np.float32)
    tracks[0, : 9000 * 16] = 0.5
    tracks[1, 10000 * 16 :] = 0.5
    direct = audit_tracks(snapshot, tracks, 16000, policy.audit)
    assert direct.mapping == (4, 7)
    swapped = audit_tracks(snapshot, tracks[[1, 0]], 16000, policy.audit)
    assert swapped.mapping == (7, 4)


def test_audit_requires_both_speakers(policy):
    snapshot = ChunkDiarization(
        (ChunkSegment(0, 9000, 4), ChunkSegment(10000, 19000, 7))
    )
    tracks = np.zeros((2, 19000 * 16), dtype=np.float32)
    tracks[0, : 9000 * 16] = 0.5
    with pytest.raises(QualityRejection):
        audit_tracks(snapshot, tracks, 16000, policy.audit)
