import pytest

from voice_pipeline_chunk_contracts import TaggedTextError, parse_text_with_audio_tags


@pytest.mark.parametrize(
    ("tagged", "plain", "tags"),
    [
        ("[thoughtful]Correct.", "Correct.", ("[thoughtful]",)),
        (
            "[sighs][thoughtful]新生成的[sighs]台词",
            "新生成的台词",
            ("[sighs]", "[thoughtful]", "[sighs]"),
        ),
        ("I [sighs] don't know.", "I don't know.", ("[sighs]",)),
        ("No tags here.", "No tags here.", ()),
    ],
)
def test_parse_text_with_audio_tags(tagged, plain, tags):
    parsed = parse_text_with_audio_tags(tagged)
    assert parsed.text_with_audio_tags == tagged
    assert parsed.text == plain
    assert parsed.tags == tags


@pytest.mark.parametrize(
    ("tagged", "code"),
    [
        ("[unknown]Hello.", "unknown_audio_tag"),
        ("[sighs[thoughtful]]Hello.", "malformed_audio_tag"),
        ("[sighs Hello.", "malformed_audio_tag"),
        ("Hello.]", "malformed_audio_tag"),
        (
            "[sighs][thoughtful][calm]Hello.",
            "too_many_consecutive_audio_tags",
        ),
        (
            "[sighs] [thoughtful] [calm]Hello.",
            "too_many_consecutive_audio_tags",
        ),
    ],
)
def test_parse_text_with_audio_tags_rejects_invalid_input(tagged, code):
    with pytest.raises(TaggedTextError) as caught:
        parse_text_with_audio_tags(tagged)
    assert caught.value.code == code
