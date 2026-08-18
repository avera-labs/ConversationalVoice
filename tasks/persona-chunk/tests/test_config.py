import pytest

from voice_pipeline_persona_chunk.config import (
    ConfigurationError,
    TaskPolicy,
    load_settings,
)


def test_default_configuration_is_openrouter_only(policy):
    assert policy.openrouter.model == "xiaomi/mimo-v2.5"
    assert policy.config_version == "persona-v1"
    assert policy.audio.sample_rate_hz == 16000
    assert policy.audio.bitrate_kbps == 48


def test_api_key_is_required():
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        load_settings(
            {
                "DATABASE_URL": "x",
                "CELERY_BROKER_URL": "x",
                "S3_BUCKET": "x",
                "S3_REGION": "x",
            }
        )


@pytest.mark.parametrize("prefix", ["*", "?", "[abc]", "unsafe.prefix", "path/name"])
def test_workspace_prefix_rejects_glob_and_path_characters(prefix):
    with pytest.raises(ValueError, match="safe name prefix"):
        TaskPolicy(workspace_prefix=prefix, error_max_length=512)
