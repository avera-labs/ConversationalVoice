import pytest

from voice_pipeline_transcribe_chunk.utterances import (
    DecodedWord,
    Word,
    build_utterances,
    normalize_words,
)


def test_normalize_rebases_and_caps_overlong_word(policy):
    words = normalize_words(
        [DecodedWord(" hello ", 0.1, 3.1, 0.75)],
        offset_ms=500,
        duration_ms=5000,
        policy=policy.utterance,
    )
    assert words == [Word(600, 900, "hello", 0.75)]


def test_confidence_must_be_finite(policy):
    with pytest.raises(ValueError):
        normalize_words(
            [DecodedWord("hello", 0.1, 0.2, float("nan"))],
            offset_ms=0,
            duration_ms=1000,
            policy=policy.utterance,
        )


def test_utterances_follow_punctuation_and_abbreviation_rules(policy):
    words = [
        Word(0, 100, "Dr.", 0.8),
        Word(120, 200, "Smith", 1.0),
        Word(220, 300, "left.", 0.9),
        Word(400, 500, "Next", 0.7),
    ]
    utterances = build_utterances(words, policy.utterance)
    assert [item.text for item in utterances] == ["Dr. Smith left.", "Next"]
    assert utterances[0].confidence == pytest.approx(0.9)
