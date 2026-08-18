from pathlib import Path

import pytest

from voice_pipeline_quality_filter_audio_part.config import (
    ConfigurationError,
    DEFAULT_MUSIC_MODEL_CACHE_DIR,
    QualityPolicy,
    load_settings,
)


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "DATABASE_URL": "postgresql://db/example",
        "CELERY_BROKER_URL": "redis://broker/0",
        "S3_BUCKET": "example-bucket",
        "S3_REGION": "us-east-1",
        "MUSIC_MODEL_CACHE_DIR": str(tmp_path),
    }


def test_defaults_load_with_deployment_environment(tmp_path: Path) -> None:
    settings = load_settings(environment(tmp_path))
    assert settings.policy.quality.min_snr_db == 10.0
    assert settings.policy.planner.max_planning_window_ms == 60000
    assert settings.policy.music.sample_rate == 22050
    assert settings.environment.music_model_cache_dir == tmp_path


def test_missing_environment_is_rejected(tmp_path: Path) -> None:
    values = environment(tmp_path)
    del values["DATABASE_URL"]
    with pytest.raises(ConfigurationError):
        load_settings(values)


def test_music_cache_uses_bundled_default_when_environment_is_absent(
    tmp_path: Path,
) -> None:
    values = environment(tmp_path)
    del values["MUSIC_MODEL_CACHE_DIR"]
    settings = load_settings(values)
    assert settings.environment.music_model_cache_dir == DEFAULT_MUSIC_MODEL_CACHE_DIR


@pytest.mark.parametrize("value", ["", "   "])
def test_music_cache_uses_bundled_default_when_environment_is_blank(
    tmp_path: Path, value: str
) -> None:
    values = environment(tmp_path)
    values["MUSIC_MODEL_CACHE_DIR"] = value
    settings = load_settings(values)
    assert settings.environment.music_model_cache_dir == DEFAULT_MUSIC_MODEL_CACHE_DIR


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_non_finite_threshold_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        QualityPolicy(
            min_snr_db=value,
            music_probability_threshold=0.2,
            min_music_interval_ms=2000,
            music_gap_fill_ms=600,
            max_music_overlap_ratio=0.3,
            max_absorbable_bad_group_ms=3000,
            min_good_region_ms=20000,
        )


def test_override_file_changes_only_selected_policy(tmp_path: Path) -> None:
    override = tmp_path / "override.toml"
    override.write_text("[quality]\nmin_snr_db = 12.5\n", encoding="utf-8")
    settings = load_settings(
        environment(tmp_path), override_policy_path=override
    )
    assert settings.policy.quality.min_snr_db == 12.5
    assert settings.policy.planner.max_planning_window_ms == 60000
