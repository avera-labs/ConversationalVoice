from __future__ import annotations

from voice_pipeline_score_completed_chunks.contracts import (
    parse_group,
    parse_references,
    validate_transcript,
)


def identity(uri: str, **extra) -> dict:
    return {
        "uri": uri,
        "size_bytes": 10,
        "sha256": "a" * 64,
        **extra,
    }


def completed_results() -> dict:
    generated_tracks = [
        identity(
            f"s3://bucket/speaker-{speaker_id}.wav",
            speaker_id=speaker_id,
            diarization_speaker_id=10 + speaker_id,
            sample_rate_hz=44100,
            duration_ms=2000,
        )
        for speaker_id in range(2)
    ]
    separation_tracks = [
        identity(
            f"s3://bucket/separation-{speaker_id}.wav",
            output_slot=speaker_id,
            diarization_speaker_id=10 + speaker_id,
            sample_rate_hz=16000,
            duration_ms=2000,
        )
        for speaker_id in range(2)
    ]
    return {
        "separation": {"speaker_audio": separation_tracks},
        "transcription": {
            "language": "en",
            "input_speaker_audio": [
                {
                    key: track[key]
                    for key in (
                        "output_slot",
                        "diarization_speaker_id",
                        "uri",
                        "size_bytes",
                        "sha256",
                    )
                }
                for track in separation_tracks
            ],
            "artifacts": {"transcript": identity("s3://bucket/separation.json")},
        },
        "reconstruction": {
            "language": "en",
            "actual_duration_ms": 2000,
            "artifacts": {
                "transcript": identity("s3://bucket/reconstruction.json"),
                "speaker_audio": generated_tracks,
            },
        },
        "dialogue_extension": {
            "language": "en",
            "actual_duration_ms": 2000,
            "artifacts": {
                "transcript": identity("s3://bucket/expansion.json"),
                "speaker_audio": generated_tracks,
            },
            "inputs": {
                "speaker_references": [
                    {
                        "speaker_id": speaker_id,
                        "diarization_speaker_id": 10 + speaker_id,
                        "source": "diarization_reference",
                        "source_audio": identity(
                            f"s3://bucket/reference-{speaker_id}.wav"
                        ),
                        "selection": {
                            "timebase": "audio_part",
                            "segments": [
                                {"start_ms": 0, "end_ms": 1000, "duration_ms": 1000}
                            ],
                        },
                        "reference_audio": {
                            "sample_rate_hz": 16000,
                            "duration_ms": 1000,
                            "size_bytes": 10,
                            "sha256": "b" * 64,
                        },
                    }
                    for speaker_id in range(2)
                ]
            },
        },
    }


def test_completed_contract_extracts_three_groups_and_references() -> None:
    results = completed_results()
    separation = parse_group(results, "separation")
    reconstruction = parse_group(results, "reconstruction")
    expansion = parse_group(results, "expansion")
    references = parse_references(results)
    assert separation.tracks[0].sample_rate_hz == 16000
    assert reconstruction.tracks[1].diarization_speaker_id == 11
    assert expansion.duration_ms == 2000
    assert references[0].source == "diarization_reference"


def test_validate_transcript_checks_mapping() -> None:
    group = parse_group(completed_results(), "reconstruction")
    transcript = {
        "language": "en",
        "timebase": "reconstruction",
        "duration_ms": 2000,
        "speaker_mapping": [
            {"speaker_id": 0, "diarization_speaker_id": 10},
            {"speaker_id": 1, "diarization_speaker_id": 11},
        ],
        "utterances": [],
    }
    assert validate_transcript(transcript, group=group) is transcript


def test_validate_separation_transcript_normalizes_speaker_utterances() -> None:
    group = parse_group(completed_results(), "separation")
    transcript = {
        "language": "en",
        "timebase": "chunk",
        "speakers": [
            {
                "output_slot": 0,
                "diarization_speaker_id": 10,
                "utterances": [{"start_ms": 0, "end_ms": 1200, "text": "one"}],
            },
            {
                "output_slot": 1,
                "diarization_speaker_id": 11,
                "utterances": [{"start_ms": 300, "end_ms": 1800, "text": "two"}],
            },
        ],
    }
    normalized = validate_transcript(transcript, group=group)
    assert [item["speaker_id"] for item in normalized["utterances"]] == [0, 1]
