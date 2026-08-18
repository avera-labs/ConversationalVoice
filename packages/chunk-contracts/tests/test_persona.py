from copy import deepcopy

import pytest
from voice_pipeline_chunk_contracts import (
    ChunkContractError,
    parse_persona_document,
    parse_persona_result,
)


def speaker(tag="You enjoy having a good conversation."):
    return {
        "name": None,
        "age": None,
        "ethnicity": None,
        "gender": None,
        "tag": tag,
        "alpha": "low",
        "evidence": None,
        "primary_emotion": "neutral",
        "secondary_emotion": None,
        "emotion_intensity": "low",
        "laugh": False,
        "cry": False,
        "whisper": False,
        "shout": False,
        "sigh": False,
        "overall_tone": "calm",
    }


def wire():
    return {
        "scene": {
            "description": "Two people speak calmly.",
            "overall_tone": "calm",
            "emotion_intensity": "low",
        },
        "speakers": {"12": speaker(), "4": speaker()},
    }


def document():
    value = wire()
    return {
        "scene": value["scene"],
        "speakers": [
            {**value["speakers"][sid], "speaker_id": sid} for sid in ("12", "4")
        ],
        "usage": {
            "model": "xiaomi/mimo-v2.5",
            "in_tokens": 10,
            "out_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.001,
        },
        "schema_version": 1,
        "backend": "openrouter",
        "config_version": "persona-v1",
        "language": "en",
        "speaker_mapping": [
            {"output_slot": 0, "diarization_speaker_id": 4},
            {"output_slot": 1, "diarization_speaker_id": 12},
        ],
    }


def test_document_preserves_source_projection_and_string_sort_order():
    parsed = parse_persona_document(
        document(), speaker_mapping=(4, 12), model_id="xiaomi/mimo-v2.5"
    )
    assert list({key: parsed[key] for key in ("scene", "speakers", "usage")}) == [
        "scene",
        "speakers",
        "usage",
    ]
    assert [item["speaker_id"] for item in parsed["speakers"]] == ["12", "4"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value["speakers"][0].update(laugh=1),
        lambda value: value["speaker_mapping"].reverse(),
        lambda value: value["speakers"].reverse(),
    ],
)
def test_document_rejects_unknown_fields_types_and_order(mutation):
    value = deepcopy(document())
    mutation(value)
    with pytest.raises(ChunkContractError):
        parse_persona_document(
            value, speaker_mapping=(4, 12), model_id="xiaomi/mimo-v2.5"
        )


def test_minimal_result_binds_all_artifacts():
    result = {
        "schema_version": 1,
        "backend": "openrouter",
        "model": {"id": "xiaomi/mimo-v2.5", "config_version": "persona-v1"},
        "language": "en",
        "input_audio": {
            "uri": "s3://bucket/audio.wav",
            "size_bytes": 1,
            "sha256": "a" * 64,
        },
        "input_transcript": {
            "uri": "s3://bucket/transcript.json",
            "size_bytes": 2,
            "sha256": "b" * 64,
        },
        "artifact": {
            "uri": "s3://bucket/persona.json",
            "size_bytes": 3,
            "sha256": "c" * 64,
        },
    }
    assert (
        parse_persona_result(
            result,
            model_id="xiaomi/mimo-v2.5",
            input_audio=("s3://bucket/audio.wav", 1, "a" * 64),
            input_transcript=("s3://bucket/transcript.json", 2, "b" * 64),
            artifact=("s3://bucket/persona.json", 3, "c" * 64),
        )
        == result
    )
