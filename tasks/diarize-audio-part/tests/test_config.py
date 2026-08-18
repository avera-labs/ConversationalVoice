from pathlib import Path

import pytest

from voice_pipeline_diarize_audio_part.config import ConfigurationError, load_settings

ENVIRONMENT = {
    "DATABASE_URL": "postgresql://db.example/test",
    "CELERY_BROKER_URL": "redis://broker.example/0",
    "S3_BUCKET": "test-bucket",
    "S3_REGION": "test-region",
}


def test_default_policy() -> None:
    settings = load_settings(ENVIRONMENT)
    assert settings.policy.diarization.device == "auto"
    assert settings.policy.diarization.model == "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    assert settings.policy.speaker_reference.min_segment_ms == 4000
    assert settings.policy.speaker_reference.edge_trim_ms == 500
    assert settings.policy.speaker_reference.min_speaker_effective_ms == 4000
    assert settings.policy.speaker_reference.max_speaker_effective_ms == 30000
    assert settings.policy.speaker_reference.inter_segment_silence_ms == 500


def test_override_is_merged(tmp_path: Path) -> None:
    override = tmp_path / "override.toml"
    override.write_text('[diarization]\ndevice = "cpu"\n', encoding="utf-8")
    settings = load_settings(ENVIRONMENT, override_policy_path=override)
    assert settings.policy.diarization.device == "cpu"


def test_rejects_non_repository_model_name(tmp_path: Path) -> None:
    override = tmp_path / "override.toml"
    override.write_text('[diarization]\nmodel = "/tmp/model"\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid"):
        load_settings(ENVIRONMENT, override_policy_path=override)


def test_missing_required_environment_is_safe() -> None:
    with pytest.raises(ConfigurationError, match="S3_REGION"):
        load_settings(
            {key: value for key, value in ENVIRONMENT.items() if key != "S3_REGION"}
        )


def test_rejects_inconsistent_speaker_reference_limits(tmp_path: Path) -> None:
    override = tmp_path / "override.toml"
    override.write_text(
        "[speaker_reference]\n"
        "min_speaker_effective_ms = 5000\n"
        "max_speaker_effective_ms = 4999\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="invalid"):
        load_settings(ENVIRONMENT, override_policy_path=override)
