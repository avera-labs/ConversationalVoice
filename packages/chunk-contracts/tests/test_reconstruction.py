import pytest

from voice_pipeline_chunk_contracts import (
    ChunkContractError,
    parse_reconstruction_transcript,
)


def test_reconstruction_transcript_contract():
    document = {
        "schema_version": 1,
        "language": "en",
        "timebase": "reconstruction",
        "source_duration_ms": 1000,
        "duration_ms": 900,
        "speaker_mapping": [
            {"speaker_id": 0, "diarization_speaker_id": 4},
            {"speaker_id": 1, "diarization_speaker_id": 7},
        ],
        "utterances": [
            {
                "utterance_index": 0,
                "speaker_id": 0,
                "diarization_speaker_id": 4,
                "speaker_utterance_index": 0,
                "text": "Hello.",
                "text_with_audio_tags": "[calm]Hello.",
                "instruction": "Speak calmly and clearly.",
                "confidence": 0.9,
                "source_start_ms": 100,
                "source_end_ms": 800,
                "start_ms": 100,
                "end_ms": 900,
                "relation": "leading",
                "anchor_utterance_index": None,
            }
        ],
    }
    assert (
        parse_reconstruction_transcript(
            document, speaker_mapping=(4, 7), source_duration_ms=1000
        )
        == document
    )


def test_chinese_reconstruction_transcript_contract():
    document = {
        "schema_version": 1,
        "language": "zh",
        "timebase": "reconstruction",
        "source_duration_ms": 1000,
        "duration_ms": 900,
        "speaker_mapping": [
            {"speaker_id": 0, "diarization_speaker_id": 4},
            {"speaker_id": 1, "diarization_speaker_id": 7},
        ],
        "utterances": [
            {
                "utterance_index": 0,
                "speaker_id": 0,
                "diarization_speaker_id": 4,
                "speaker_utterance_index": 0,
                "text": "你好。",
                "text_with_audio_tags": "[calm]你好。",
                "instruction": "平静而清晰地说。",
                "confidence": 0.9,
                "source_start_ms": 100,
                "source_end_ms": 800,
                "start_ms": 100,
                "end_ms": 900,
                "relation": "leading",
                "anchor_utterance_index": None,
            }
        ],
    }
    assert (
        parse_reconstruction_transcript(
            document,
            speaker_mapping=(4, 7),
            source_duration_ms=1000,
            expected_language="zh",
        )
        == document
    )

    with pytest.raises(ChunkContractError, match="identity"):
        parse_reconstruction_transcript(
            document,
            speaker_mapping=(4, 7),
            source_duration_ms=1000,
        )
