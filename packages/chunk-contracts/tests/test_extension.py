from copy import deepcopy

import pytest
from voice_pipeline_chunk_contracts import (
    AlignedTextUnit,
    ChunkContractError,
    build_segment_word_alignment,
    offset_word_alignment,
    parse_dialogue_extension_document,
    parse_dialogue_extension_transcript,
)


def timed_utterance(value, start_ms, end_ms):
    unit = "".join(character for character in value["text"] if character.isalnum())
    segment = build_segment_word_alignment(
        value["text_with_audio_tags"],
        [AlignedTextUnit(unit, 0, end_ms - start_ms)] if unit else [],
        duration_ms=end_ms - start_ms,
    )
    return {
        **value,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "word_alignment": offset_word_alignment(segment, start_ms),
    }


def script(language="en"):
    chinese = language == "zh"
    return {
        "schema_version": 1,
        "backend": "openrouter",
        "model": {
            "id": "xiaomi/mimo-v2.5",
            "config_version": "dialogue-extension-v1",
        },
        "language": language,
        "target_duration_ms": 120000,
        "speaker_mapping": [
            {"speaker_id": 0, "diarization_speaker_id": 4},
            {"speaker_id": 1, "diarization_speaker_id": 7},
        ],
        "utterances": [
            {
                "utterance_index": 0,
                "speaker_id": 0,
                "text": "这正是我的意思。"
                if chinese
                else "That is exactly what I meant.",
                "text_with_audio_tags": (
                    "[thoughtful]这正是我的意思。"
                    if chinese
                    else "[thoughtful]That is exactly what I meant."
                ),
                "instruction": "温和而沉思地说。"
                if chinese
                else "Speak warmly and reflectively.",
                "type": "dialogue",
                "placement": "sequential",
            },
            {
                "utterance_index": 1,
                "speaker_id": 1,
                "text": "对。" if chinese else "Yeah.",
                "text_with_audio_tags": "对。" if chinese else "Yeah.",
                "instruction": "快速地表示赞同。"
                if chinese
                else "Give a quick agreement.",
                "type": "backchannel",
                "placement": "overlap_previous",
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
            timed_utterance(value["utterances"][0], 0, 1800),
            timed_utterance(value["utterances"][1], 1400, 2200),
        ],
    }
    assert (
        parse_dialogue_extension_transcript(
            transcript, script=parsed, speaker_mapping=(4, 7)
        )["duration_ms"]
        == 2200
    )


def test_chinese_script_and_timed_transcript_accept_canonical_documents():
    value = script("zh")
    with pytest.raises(ChunkContractError, match="identity"):
        parse_dialogue_extension_document(
            value,
            speaker_mapping=(4, 7),
            model_id="xiaomi/mimo-v2.5",
            target_duration_ms=120000,
        )
    parsed = parse_dialogue_extension_document(
        value,
        speaker_mapping=(4, 7),
        model_id="xiaomi/mimo-v2.5",
        target_duration_ms=120000,
        expected_language="zh",
    )
    transcript = {
        "schema_version": 1,
        "language": "zh",
        "timebase": "dialogue_extension",
        "duration_ms": 2200,
        "speaker_mapping": value["speaker_mapping"],
        "utterances": [
            timed_utterance(value["utterances"][0], 0, 1800),
            timed_utterance(value["utterances"][1], 1400, 2200),
        ],
    }
    assert (
        parse_dialogue_extension_transcript(
            transcript,
            script=parsed,
            speaker_mapping=(4, 7),
            expected_language="zh",
        )["language"]
        == "zh"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speaker_id", 2),
        ("text_with_audio_tags", "[unknown]That is exactly what I meant."),
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
            timed_utterance(value["utterances"][0], 0, 1800),
            timed_utterance(value["utterances"][1], 1400, 2200),
        ],
    }
    changed = deepcopy(transcript)
    changed["utterances"][1]["text"] = "Changed"
    with pytest.raises(ChunkContractError):
        parse_dialogue_extension_transcript(
            changed, script=value, speaker_mapping=(4, 7)
        )
