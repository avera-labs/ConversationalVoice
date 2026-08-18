from pathlib import Path

import pytest

from voice_pipeline_transcribe_chunk.config import ConfigurationError, load_settings

BASE = {
    "DATABASE_URL": "postgresql://example/db",
    "CELERY_BROKER_URL": "redis://example/0",
    "S3_BUCKET": "bucket",
    "S3_REGION": "us-east-1",
}


def test_default_policy_matches_pinned_contract():
    settings = load_settings(BASE)
    assert settings.policy.config_version == "parakeet-v1"
    assert settings.policy.model.revision == "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    assert settings.policy.slices.merge_gap_ms == 2000
    assert settings.policy.slices.pad_ms == 500


def test_invalid_slice_invariant_is_rejected(tmp_path: Path):
    source = (
        Path(__file__).parents[1]
        / "src/voice_pipeline_transcribe_chunk/resources/default.toml"
    )
    value = source.read_text().replace("pad_ms = 500", "pad_ms = 1001")
    custom = tmp_path / "invalid.toml"
    custom.write_text(value)
    with pytest.raises(ConfigurationError):
        load_settings({**BASE, "PARAKEET_CONFIG_FILE": str(custom)})
