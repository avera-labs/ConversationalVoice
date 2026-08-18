import pytest
from voice_pipeline_chunk_contracts import ChunkSegment

from voice_pipeline_extend_chunk.reference import (
    SpeakerReferenceUnavailable,
    longest_pure_interval,
    parse_reference_manifest,
)


def manifest():
    return {
        "schema_version": 1,
        "speakers": [
            {
                "speaker_id": speaker_id,
                "reference_audio": {
                    "uri": f"s3://bucket/speaker-{speaker_id}.wav",
                    "sample_rate_hz": 16000,
                    "size_bytes": 100,
                    "sha256": str(speaker_id + 1) * 64,
                    "segments": [{"start_ms": 0, "end_ms": 5000, "duration_ms": 5000}],
                    "effective_duration_ms": 5000,
                    "total_duration_ms": 5000,
                },
            }
            for speaker_id in (4, 7)
        ],
    }


def test_selects_references_in_chunk_output_slot_order():
    first, second = parse_reference_manifest(manifest(), expected_speaker_ids=(7, 4))
    assert first["uri"].endswith("speaker-7.wav")
    assert second["uri"].endswith("speaker-4.wav")


def test_missing_mapped_speaker_is_returned_for_per_speaker_fallback():
    first, second = parse_reference_manifest(manifest(), expected_speaker_ids=(4, 9))
    assert first["uri"].endswith("speaker-4.wav")
    assert second is None


def test_longest_pure_interval_excludes_overlap_and_caps_duration():
    segments = (
        ChunkSegment(0, 50000, 4),
        ChunkSegment(10000, 12000, 7),
        ChunkSegment(60000, 65000, 4),
    )
    assert longest_pure_interval(segments, speaker_id=4) == (12500, 42500)


def test_missing_pure_interval_is_known_quality_rejection():
    segments = (
        ChunkSegment(0, 1000, 4),
        ChunkSegment(0, 1000, 7),
    )
    with pytest.raises(SpeakerReferenceUnavailable):
        longest_pure_interval(segments, speaker_id=7)


def test_interval_must_remain_positive_after_safety_trim():
    with pytest.raises(SpeakerReferenceUnavailable):
        longest_pure_interval((ChunkSegment(0, 1000, 7),), speaker_id=7)
