from voice_pipeline_chunk_contracts import select_tts_inputs, tts_capabilities


UTTERANCE = {
    "text": "Correct.",
    "text_with_audio_tags": "[thoughtful]Correct.",
    "instruction": "Speak thoughtfully.",
}


def test_fish_audio_uses_tagged_text():
    selected = select_tts_inputs("fish-audio/s2.1-pro", UTTERANCE)
    assert selected.text == "[thoughtful]Correct."
    assert selected.instruction is None


def test_mimo_is_temporarily_unmapped_and_defaults_to_plain_text(caplog):
    selected = select_tts_inputs("mimo-v2.5-tts-voiceclone", UTTERANCE)
    assert selected.text == "Correct."
    assert selected.instruction is None
    assert "missing from TTS_MODEL_CAPABILITIES" in caplog.text
    assert "Add the model to the mapping" in caplog.text


def test_unknown_model_defaults_to_plain_text_and_warns(caplog):
    assert tts_capabilities("provider/future-model") == frozenset({"text"})
    selected = select_tts_inputs("provider/future-model", UTTERANCE)
    assert selected.text == "Correct."
    assert selected.instruction is None
    assert len(caplog.records) == 2
    assert all(record.levelname == "WARNING" for record in caplog.records)
    assert all("provider/future-model" in record.message for record in caplog.records)
