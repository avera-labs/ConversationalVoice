import pytest

from voice_pipeline_transcribe_chunk_zh.alignment import (
    AlignedUnit,
    DecodedUnit,
    attach_punctuation,
    build_utterances,
    normalize_units,
    restore_punctuation,
)


def test_each_chinese_character_keeps_its_own_timestamp(policy):
    result = normalize_units(
        [
            DecodedUnit("你", 0.1, 0.2, 0.9),
            DecodedUnit("好", 0.2, 0.4, 0.8),
        ],
        offset_ms=500,
        duration_ms=2000,
        policy=policy.utterance,
    )
    assert [(item.text, item.start_ms, item.end_ms) for item in result] == [
        ("你", 600, 700),
        ("好", 700, 900),
    ]


def test_punctuation_attaches_to_preceding_character_and_drives_utterances(policy):
    units = [
        AlignedUnit(0, 100, "你", 0.9),
        AlignedUnit(100, 200, "好", 0.8),
        AlignedUnit(300, 400, "世", 0.7),
        AlignedUnit(400, 500, "界", 0.6),
    ]
    punctuated = attach_punctuation(units, "你好，世界！")
    assert [item.text for item in punctuated] == ["你", "好，", "世", "界！"]
    utterances = build_utterances(punctuated, policy.utterance)
    assert [item.text for item in utterances] == ["你好，世界！"]
    assert utterances[0].start_ms == 0 and utterances[0].end_ms == 500


def test_ascii_period_after_english_word_is_attached_and_splits_utterance(policy):
    units = [
        AlignedUnit(0, 100, "computing", 0.9),
        AlignedUnit(100, 200, "devices", 0.8),
        AlignedUnit(200, 300, "或", 0.9),
    ]

    punctuated = attach_punctuation(units, "computing devices.或")

    assert [item.text for item in punctuated] == ["computing", "devices.", "或"]
    assert [item.text for item in build_utterances(punctuated, policy.utterance)] == [
        "computingdevices.",
        "或",
    ]


@pytest.mark.parametrize("value", ["你坏。", "你。", "好你。"])
def test_punctuation_model_cannot_rewrite_drop_or_reorder_text(value):
    units = [AlignedUnit(0, 100, "你", 0.9), AlignedUnit(100, 200, "好", 0.8)]
    with pytest.raises(ValueError):
        attach_punctuation(units, value)


def test_multi_character_chinese_model_unit_is_rejected(policy):
    with pytest.raises(ValueError, match="decoded alignment unit is invalid"):
        normalize_units(
            [DecodedUnit("你好", 0.1, 0.4, 0.9)],
            offset_ms=0,
            duration_ms=1000,
            policy=policy.utterance,
        )


def test_punctuation_inference_is_split_into_bounded_character_batches():
    units = [
        AlignedUnit(index * 100, index * 100 + 100, char, 0.9)
        for index, char in enumerate("你好世界")
    ]

    class Punctuation:
        def __init__(self):
            self.inputs = []

        def restore(self, text):
            self.inputs.append(text)
            return text + "。"

    punctuation = Punctuation()
    result = restore_punctuation(units, punctuation, max_chars=2)
    assert punctuation.inputs == ["你好", "世界"]
    assert [item.text for item in result] == ["你", "好。", "世", "界。"]
