from copy import deepcopy

import pytest
from voice_pipeline_chunk_contracts import (
    ChunkContractError,
    parse_dialogue_extension_document,
    parse_dialogue_extension_transcript,
)


def script():
    return {
        "schema_version": 1,
        "backend": "openrouter",
        "model": {
            "id": "xiaomi/mimo-v2.5",
            "config_version": "dialogue-extension-v1",
        },
        "language": "en",
        "target_duration_ms": 120000,
        "speaker_mapping": [
            {"speaker_id": 0, "diarization_speaker_id": 4},
            {"speaker_id": 1, "diarization_speaker_id": 7},
        ],
        "utterances": [
            {
                "utterance_index": 0,
                "speaker_id": 0,
                "text": "That is exactly what I meant.",
                "tone": "warm and reflective",
                "type": "dialogue",
                "placement": "sequential",
                "audio_tags": ["[thoughtful]"],
            },
            {
                "utterance_index": 1,
                "speaker_id": 1,
                "text": "Yeah.",
                "tone": "quick agreement",
                "type": "backchannel",
                "placement": "overlap_previous",
                "audio_tags": [],
            },
        ],
        "usage": {
            "model": "xiaomi/mimo-v2.5",
            "in_tokens": 10,
            "out_tokens": 20,
            "total_tokens": 30,
            "cost_usd": 0.001,
        },
    }


def test_script_and_timed_transcript_accept_canonical_documents():
    value = script()
    parsed = parse_dialogue_extension_document(
        value,
        speaker_mapping=(4, 7),
        model_id="xiaomi/mimo-v2.5",
        target_duration_ms=120000,
    )
    transcript = {
        "schema_version": 1,
        "language": "en",
        "timebase": "dialogue_extension",
        "duration_ms": 2200,
        "speaker_mapping": value["speaker_mapping"],
        "utterances": [
            {**value["utterances"][0], "start_ms": 0, "end_ms": 1800},
            {**value["utterances"][1], "start_ms": 1400, "end_ms": 2200},
        ],
    }
    assert (
        parse_dialogue_extension_transcript(
            transcript, script=parsed, speaker_mapping=(4, 7)
        )["duration_ms"]
        == 2200
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speaker_id", 2),
        ("audio_tags", ["[unknown]"]),
        ("text", "[laughs] hidden tag"),
    ],
)
def test_script_rejects_invalid_utterance_values(field, value):
    document = script()
    document["utterances"][0][field] = value
    with pytest.raises(ChunkContractError):
        parse_dialogue_extension_document(
            document,
            speaker_mapping=(4, 7),
            model_id="xiaomi/mimo-v2.5",
            target_duration_ms=120000,
        )


def test_script_rejects_dialogue_overlap():
    document = script()
    document["utterances"][1]["type"] = "dialogue"
    with pytest.raises(ChunkContractError):
        parse_dialogue_extension_document(
            document,
            speaker_mapping=(4, 7),
            model_id="xiaomi/mimo-v2.5",
            target_duration_ms=120000,
        )


def test_script_enforces_configured_utterance_count():
    document = script()
    with pytest.raises(ChunkContractError, match="utterance count"):
        parse_dialogue_extension_document(
            document,
            speaker_mapping=(4, 7),
            model_id="xiaomi/mimo-v2.5",
            target_duration_ms=120000,
            min_utterances=8,
            max_utterances=40,
        )


def test_transcript_rejects_script_mutation():
    value = script()
    transcript = {
        "schema_version": 1,
        "language": "en",
        "timebase": "dialogue_extension",
        "duration_ms": 2200,
        "speaker_mapping": value["speaker_mapping"],
        "utterances": [
            {**value["utterances"][0], "start_ms": 0, "end_ms": 1800},
            {**value["utterances"][1], "start_ms": 1400, "end_ms": 2200},
        ],
    }
    changed = deepcopy(transcript)
    changed["utterances"][1]["text"] = "Changed"
    with pytest.raises(ChunkContractError):
        parse_dialogue_extension_transcript(
            changed, script=value, speaker_mapping=(4, 7)
        )
