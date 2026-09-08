from voice_pipeline_reconstruct_chunk.config import TtsPolicy


def test_default_models_and_audio_policy(policy):
    assert policy.audio_tags.model == "google/gemini-3.7-flash"
    assert policy.audio_tags.max_attempts == 3
    assert policy.audio_tags.reasoning_effort == "low"
    assert policy.audio_tags.max_tokens == 2048
    assert policy.forced_alignment.repo_id == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert len(policy.forced_alignment.revision) == 40
    assert policy.tts.model == "fish-audio/s2.1-pro"
    assert policy.audio.reference_silence_ms == 1000
    assert policy.audio.input_sample_rate_hz == 16000
    assert policy.audio.output_sample_rate_hz == 44100
    assert policy.audio_tags.require_parameters is True
    assert policy.audio_tags.allow_fallbacks is True


def test_tts_model_accepts_models_outside_the_capability_mapping(policy):
    values = policy.tts.model_dump()
    values["model"] = "provider/plain-tts"

    assert TtsPolicy.model_validate(values).model == "provider/plain-tts"
