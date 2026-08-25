from pathlib import Path

import pytest

from voice_pipeline_split_raw_audio_into_parts.config import (
    ConfigurationError,
    load_application_settings,
    load_environment_settings,
    load_policy_settings,
)


ENVIRONMENT = {
    "DATABASE_URL": "postgresql://example.test/database",
    "CELERY_BROKER_URL": "redis://example.test:6379/0",
    "S3_BUCKET": "test-bucket",
    "S3_REGION": "test-region-1",
    "HF_TOKEN": "<hf-token>",
}


def test_packaged_policy_defaults() -> None:
    policy = load_policy_settings()

    assert policy.vad.model == "pyannote/segmentation-3.0"
    assert policy.vad.device == "cpu"
    assert policy.windowing.gap_threshold_ms == 15_000
    assert policy.windowing.min_window_ms == 20_000
    assert policy.windowing.max_window_ms == 900_000
    assert policy.windowing.pad_before_ms == 250
    assert policy.windowing.pad_after_ms == 250


def test_policy_override_merges_only_supplied_values(tmp_path: Path) -> None:
    override = tmp_path / "override.toml"
    override.write_text(
        "[windowing]\ngap_threshold_ms = 12000\n",
        encoding="utf-8",
    )

    policy = load_policy_settings(override_path=override)

    assert policy.windowing.gap_threshold_ms == 12_000
    assert policy.windowing.min_window_ms == 20_000
    assert policy.vad.model == "pyannote/segmentation-3.0"


@pytest.mark.parametrize(
    "document",
    [
        "[unknown]\nvalue = 1\n",
        "[vad]\ndevice = \"tpu\"\n",
        "[windowing]\ngap_threshold_ms = true\n",
        "[windowing]\nmin_window_ms = 1000\nmax_window_ms = 999\n",
    ],
)
def test_invalid_policy_override_fails_fast(
    tmp_path: Path,
    document: str,
) -> None:
    override = tmp_path / "invalid.toml"
    override.write_text(document, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Policy configuration is invalid"):
        load_policy_settings(override_path=override)


def test_environment_uses_shared_names_and_masks_token() -> None:
    settings = load_environment_settings(ENVIRONMENT)

    assert settings.database_url == ENVIRONMENT["DATABASE_URL"]
    assert settings.celery_broker_url == ENVIRONMENT["CELERY_BROKER_URL"]
    assert settings.s3_bucket == "test-bucket"
    assert settings.s3_endpoint_url is None
    assert settings.hf_token.get_secret_value() == "<hf-token>"
    assert "<hf-token>" not in repr(settings)


def test_missing_environment_values_fail_without_exposing_values() -> None:
    environment = dict(ENVIRONMENT)
    environment.pop("DATABASE_URL")

    with pytest.raises(ConfigurationError) as error:
        load_environment_settings(environment)

    assert str(error.value) == "Missing required environment variables: DATABASE_URL."
    assert "<hf-token>" not in str(error.value)


def test_application_settings_combine_policy_and_environment() -> None:
    settings = load_application_settings(ENVIRONMENT)

    assert settings.policy.windowing.max_window_ms == 900_000
    assert settings.environment.s3_region == "test-region-1"
