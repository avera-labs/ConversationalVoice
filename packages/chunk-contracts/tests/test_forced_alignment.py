import pytest

from voice_pipeline_chunk_contracts import (
    AlignedTextUnit,
    ChunkContractError,
    build_segment_word_alignment,
    fit_segment_word_alignment,
    offset_word_alignment,
    validate_utterance_word_alignment,
)


def test_build_alignment_inserts_zero_duration_tags_at_text_positions():
    alignment = build_segment_word_alignment(
        "[sighs]Hello [thoughtful]world.[sighs]",
        [AlignedTextUnit("Hello", 100, 400), AlignedTextUnit("world", 500, 900)],
        duration_ms=1000,
    )

    assert [(item["type"], item["text"]) for item in alignment] == [
        ("audio_tag", "[sighs]"),
        ("word", "Hello"),
        ("audio_tag", "[thoughtful]"),
        ("word", "world"),
        ("audio_tag", "[sighs]"),
    ]
    tags = [item for item in alignment if item["type"] == "audio_tag"]
    assert [(item["text_start"], item["start_ms"], item["end_ms"]) for item in tags] == [
        (0, 100, 100),
        (6, 500, 500),
        (12, 900, 900),
    ]


def test_build_alignment_places_mid_word_tag_without_duration():
    alignment = build_segment_word_alignment(
        "hel[sighs]lo", [AlignedTextUnit("hello", 100, 600)], duration_ms=700
    )

    assert alignment[1] == {
        "item_index": 1,
        "type": "audio_tag",
        "text": "[sighs]",
        "text_start": 3,
        "text_end": 3,
        "start_ms": 400,
        "end_ms": 400,
    }


def test_tag_immediately_after_word_uses_previous_word_end():
    alignment = build_segment_word_alignment(
        "Hello[sighs] world",
        [AlignedTextUnit("Hello", 100, 400), AlignedTextUnit("world", 600, 900)],
        duration_ms=1000,
    )

    tag = next(item for item in alignment if item["type"] == "audio_tag")
    assert tag["start_ms"] == tag["end_ms"] == 400


def test_repeated_chinese_tags_keep_original_word_level_positions():
    text = "新生成的台词"
    units = [
        AlignedTextUnit(character, index * 100, (index + 1) * 100)
        for index, character in enumerate(text)
    ]

    alignment = build_segment_word_alignment(
        "[sighs][thoughtful]新生成的[sighs]台词",
        units,
        duration_ms=700,
    )

    assert [item["text"] for item in alignment] == [
        "[sighs]",
        "[thoughtful]",
        "新",
        "生",
        "成",
        "的",
        "[sighs]",
        "台",
        "词",
    ]
    assert alignment[6]["text_start"] == alignment[6]["text_end"] == 4
    assert alignment[6]["start_ms"] == alignment[6]["end_ms"] == 400


def test_global_offset_and_contract_validation():
    segment = build_segment_word_alignment(
        "[calm]Hello.", [AlignedTextUnit("Hello", 20, 380)], duration_ms=500
    )
    global_alignment = offset_word_alignment(segment, 1200)

    assert (
        validate_utterance_word_alignment(
            global_alignment,
            text_with_audio_tags="[calm]Hello.",
            start_ms=1200,
            end_ms=1700,
        )
        == global_alignment
    )

    broken = [dict(item) for item in global_alignment]
    broken[0]["end_ms"] += 1
    with pytest.raises(ChunkContractError):
        validate_utterance_word_alignment(
            broken,
            text_with_audio_tags="[calm]Hello.",
            start_ms=1200,
            end_ms=1700,
        )


def test_fit_segment_alignment_handles_frame_rounding_difference():
    alignment = build_segment_word_alignment(
        "Hello.[sighs]",
        [AlignedTextUnit("Hello", 100, 501)],
        duration_ms=501,
    )

    fitted = fit_segment_word_alignment(
        "Hello.[sighs]", alignment, duration_ms=500
    )

    assert fitted[-2]["end_ms"] == 500
    assert fitted[-1]["start_ms"] == fitted[-1]["end_ms"] == 500
