import json

import pytest
from voice_pipeline_diarization_artifact import Segment

from voice_pipeline_diarize_audio_part.config import SpeakerReferencePolicy
from voice_pipeline_diarize_audio_part.speaker_reference import (
    ManifestSpeaker,
    ReferenceAudio,
    ReferenceSegment,
    SpeakerReferenceError,
    SpeakerReferenceManifest,
    plan_speaker_references,
    pure_intervals,
)


def segment(speaker: int, start_ms: int, end_ms: int) -> Segment:
    return Segment(
        speaker=speaker,
        start=start_ms / 1000,
        end=end_ms / 1000,
        duration=(end_ms - start_ms) / 1000,
    )


def policy(**overrides: int) -> SpeakerReferencePolicy:
    values = {
        "min_segment_ms": 4000,
        "edge_trim_ms": 500,
        "min_speaker_effective_ms": 4000,
        "max_speaker_effective_ms": 30000,
        "inter_segment_silence_ms": 500,
    }
    values.update(overrides)
    return SpeakerReferencePolicy(**values)


def test_pure_intervals_remove_only_actual_cross_speaker_overlap() -> None:
    result = pure_intervals(
        (
            segment(0, 0, 10000),
            segment(1, 4000, 6000),
        )
    )
    assert [(item.speaker_id, item.start_ms, item.end_ms) for item in result] == [
        (0, 0, 4000),
        (0, 6000, 10000),
    ]


def test_pure_intervals_merge_same_speaker_overlap_and_adjacency() -> None:
    result = pure_intervals(
        (
            segment(2, 0, 5000),
            segment(2, 3000, 8000),
            segment(2, 8000, 10000),
        )
    )
    assert [(item.speaker_id, item.start_ms, item.end_ms) for item in result] == [
        (2, 0, 10000)
    ]


def test_safety_trim_precedes_strict_segment_threshold() -> None:
    assert plan_speaker_references((segment(0, 0, 5000),), policy()) == ()
    accepted = plan_speaker_references((segment(0, 0, 5001),), policy())
    assert accepted[0].segments == (ReferenceSegment(500, 4501),)
    assert accepted[0].effective_duration_ms == 4001


def test_longest_first_collection_and_final_cap_truncation() -> None:
    result = plan_speaker_references(
        (
            segment(0, 0, 12000),
            segment(0, 20000, 45000),
        ),
        policy(),
    )
    assert result[0].segments == (
        ReferenceSegment(20500, 44500),
        ReferenceSegment(500, 6500),
    )
    assert result[0].effective_duration_ms == 30000


def test_speaker_below_configured_effective_minimum_is_ignored() -> None:
    result = plan_speaker_references(
        (segment(0, 0, 6000),),
        policy(min_speaker_effective_ms=6000),
    )
    assert result == ()


def test_speaker_plans_are_sorted_by_numeric_id() -> None:
    result = plan_speaker_references(
        (
            segment(9, 0, 6000),
            segment(2, 10000, 16000),
        ),
        policy(),
    )
    assert [item.speaker_id for item in result] == [2, 9]


def test_empty_manifest_has_canonical_bytes() -> None:
    manifest = SpeakerReferenceManifest(())
    assert json.loads(manifest.to_json_bytes()) == {
        "schema_version": 1,
        "speakers": [],
    }
    assert manifest.to_json_bytes().endswith(b"\n")


def test_manifest_exact_shape() -> None:
    manifest = SpeakerReferenceManifest(
        (
            ManifestSpeaker(
                speaker_id=0,
                reference_audio=ReferenceAudio(
                    uri="s3://bucket/path/speaker-0.wav",
                    sample_rate_hz=16000,
                    size_bytes=100,
                    sha256="0" * 64,
                    segments=(ReferenceSegment(500, 5001),),
                    effective_duration_ms=4501,
                    total_duration_ms=4501,
                ),
            ),
        )
    )
    assert set(manifest.to_dict()) == {"schema_version", "speakers"}
    speaker = manifest.to_dict()["speakers"][0]
    assert set(speaker) == {"speaker_id", "reference_audio"}
    assert set(speaker["reference_audio"]) == {
        "uri",
        "sample_rate_hz",
        "size_bytes",
        "sha256",
        "segments",
        "effective_duration_ms",
        "total_duration_ms",
    }


def test_manifest_rejects_unsorted_or_duplicate_speakers() -> None:
    audio = ReferenceAudio(
        uri="s3://bucket/path.wav",
        sample_rate_hz=16000,
        size_bytes=100,
        sha256="0" * 64,
        segments=(ReferenceSegment(0, 4001),),
        effective_duration_ms=4001,
        total_duration_ms=4001,
    )
    with pytest.raises(SpeakerReferenceError):
        SpeakerReferenceManifest((ManifestSpeaker(1, audio), ManifestSpeaker(0, audio)))
