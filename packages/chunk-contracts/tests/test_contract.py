from copy import deepcopy

import pytest
from voice_pipeline_chunk_contracts import (
    ChunkContractError,
    build_chunk_diarization,
    parse_chunk_diarization,
    parse_separation_result,
    parse_transcription_artifact,
    parse_transcription_result,
)
from voice_pipeline_diarization_artifact import DiarizationTurn


def snapshot():
    return build_chunk_diarization(
        (DiarizationTurn(90, 5000, 4), DiarizationTurn(6000, 15000, 7)),
        start_ms=100,
        end_ms=12100,
    )


def separation():
    return {
        "schema_version": 1,
        "backend": "dialogue_sidon",
        "model": {
            "repo_id": "sarulab-speech/DialogueSidon",
            "revision": "a" * 40,
            "config_version": "sidon-v1",
            "inference_steps": 100,
        },
        "input_audio": {
            "sample_rate_hz": 16000,
            "duration_ms": 12000,
            "size_bytes": 10,
            "sha256": "b" * 64,
        },
        "speaker_audio": [
            {
                "output_slot": 0,
                "diarization_speaker_id": 4,
                "uri": "s3://bucket/speaker-0.wav",
                "sample_rate_hz": 16000,
                "duration_ms": 12000,
                "size_bytes": 11,
                "sha256": "c" * 64,
            },
            {
                "output_slot": 1,
                "diarization_speaker_id": 7,
                "uri": "s3://bucket/speaker-1.wav",
                "sample_rate_hz": 16000,
                "duration_ms": 12000,
                "size_bytes": 12,
                "sha256": "d" * 64,
            },
        ],
        "audit": {
            "verdict": "ok",
            "reference_speaker_id": 4,
            "consistent_relation": "direct",
        },
    }


def test_snapshot_clips_and_preserves_speaker_ids():
    value = snapshot()
    assert value.speaker_ids == (4, 7)
    assert value.segments[0].start_ms == 0
    assert parse_chunk_diarization(value.to_dict(), duration_ms=12000) == value


def test_separation_mapping_is_strict():
    result = parse_separation_result(
        separation(),
        duration_ms=12000,
        speaker_ids=(4, 7),
        input_size_bytes=10,
        input_sha256="b" * 64,
        output_uris=("s3://bucket/speaker-0.wav", "s3://bucket/speaker-1.wav"),
    )
    assert [item.diarization_speaker_id for item in result.speaker_audio] == [4, 7]


@pytest.mark.parametrize("extra", ["windows", "seed", "alignment", "stitching"])
def test_minimal_v1_rejects_diagnostics(extra):
    value = separation()
    value[extra] = {}
    with pytest.raises(ChunkContractError):
        parse_separation_result(
            value,
            duration_ms=12000,
            speaker_ids=(4, 7),
            input_size_bytes=10,
            input_sha256="b" * 64,
            output_uris=("s3://bucket/speaker-0.wav", "s3://bucket/speaker-1.wav"),
        )


def test_swapped_relation_requires_swapped_mapping():
    value = separation()
    value["audit"]["consistent_relation"] = "swapped"
    value["speaker_audio"][0]["diarization_speaker_id"] = 7
    value["speaker_audio"][1]["diarization_speaker_id"] = 4
    parse_separation_result(
        value,
        duration_ms=12000,
        speaker_ids=(4, 7),
        input_size_bytes=10,
        input_sha256="b" * 64,
        output_uris=("s3://bucket/speaker-0.wav", "s3://bucket/speaker-1.wav"),
    )
    broken = deepcopy(value)
    broken["speaker_audio"].reverse()
    with pytest.raises(ChunkContractError):
        parse_separation_result(
            broken,
            duration_ms=12000,
            speaker_ids=(4, 7),
            input_size_bytes=10,
            input_sha256="b" * 64,
            output_uris=("s3://bucket/speaker-0.wav", "s3://bucket/speaker-1.wav"),
        )


def transcription_artifact(kind="transcript"):
    key = "utterances" if kind == "transcript" else "words"
    index = "utterance_index" if kind == "transcript" else "word_index"
    return {
        "schema_version": 1,
        "backend": "parakeet_tdt",
        "model": {
            "repo_id": "nvidia/parakeet-tdt-0.6b-v3",
            "revision": "e" * 40,
            "config_version": "parakeet-v1",
        },
        "language": "en",
        "timebase": "chunk",
        "speakers": [
            {
                "output_slot": 0,
                "diarization_speaker_id": 4,
                key: [
                    {
                        index: 0,
                        "start_ms": 100,
                        "end_ms": 200,
                        "text": "Hello.",
                        "confidence": 0.9,
                    }
                ],
            },
            {"output_slot": 1, "diarization_speaker_id": 7, key: []},
        ],
    }


def test_transcription_artifacts_accept_canonical_empty_speaker():
    for kind in ("transcript", "word_alignment"):
        parse_transcription_artifact(
            transcription_artifact(kind),
            kind=kind,
            duration_ms=1000,
            speaker_mapping=(4, 7),
        )


def test_transcription_artifact_rejects_unknown_fields():
    value = transcription_artifact()
    value["audit"] = {}
    with pytest.raises(ChunkContractError):
        parse_transcription_artifact(
            value, kind="transcript", duration_ms=1000, speaker_mapping=(4, 7)
        )


def test_minimal_transcription_result_binds_inputs_and_artifacts():
    parsed_separation = parse_separation_result(
        separation(),
        duration_ms=12000,
        speaker_ids=(4, 7),
        input_size_bytes=10,
        input_sha256="b" * 64,
        output_uris=("s3://bucket/speaker-0.wav", "s3://bucket/speaker-1.wav"),
    )
    value = {
        "schema_version": 1,
        "backend": "parakeet_tdt",
        "model": {
            "repo_id": "nvidia/parakeet-tdt-0.6b-v3",
            "revision": "e" * 40,
            "config_version": "parakeet-v1",
        },
        "language": "en",
        "input_speaker_audio": [
            {
                "output_slot": item.output_slot,
                "diarization_speaker_id": item.diarization_speaker_id,
                "uri": item.uri,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in parsed_separation.speaker_audio
        ],
        "artifacts": {
            "transcript": {
                "uri": "s3://bucket/transcript.json",
                "size_bytes": 1,
                "sha256": "f" * 64,
            },
            "word_alignment": {
                "uri": "s3://bucket/word_alignment.json",
                "size_bytes": 2,
                "sha256": "0" * 64,
            },
        },
    }
    parse_transcription_result(
        value,
        speaker_audio=parsed_separation.speaker_audio,
        artifact_uris=(
            "s3://bucket/transcript.json",
            "s3://bucket/word_alignment.json",
        ),
        artifact_metadata=((1, "f" * 64), (2, "0" * 64)),
    )
