import pytest

from voice_pipeline_extend_chunk.config import (
    ConfigurationError,
    FishAudioPolicy,
    load_settings,
)


def test_default_policy_targets_two_minutes(settings):
    assert settings.policy.openrouter.model == "google/gemini-3.7-flash"
    assert settings.policy.openrouter.max_attempts == 3
    assert settings.policy.dialogue.target_duration_seconds == 120
    assert settings.policy.dialogue.target_words == 300
    assert settings.policy.config_version == "dialogue-extension-v3-duration-aware-overlap"
    assert settings.policy.timeline.overlap_min_anchor_fraction == 0.6
    assert settings.policy.fish_audio.model == "fish-audio/s2.1-pro"
    assert settings.policy.fish_audio.transcription_model == "fish-audio/transcribe-1"
    assert (
        settings.policy.forced_alignment.repo_id
        == "Qwen/Qwen3-ForcedAligner-0.6B"
    )
    assert len(settings.policy.forced_alignment.revision) == 40


def test_openrouter_key_is_required_for_dialogue_and_speech():
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        load_settings(
            {
                "DATABASE_URL": "postgresql://example/db",
                "CELERY_BROKER_URL": "redis://example/0",
                "S3_BUCKET": "bucket",
                "S3_REGION": "us-east-1",
            }
        )


def test_tts_model_accepts_models_outside_the_capability_mapping(settings):
    values = settings.policy.fish_audio.model_dump()
    values["model"] = "provider/plain-tts"

    assert FishAudioPolicy.model_validate(values).model == "provider/plain-tts"
