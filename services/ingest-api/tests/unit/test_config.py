from __future__ import annotations

from pathlib import Path

import pytest

from voice_pipeline_ingest_api.config import (
    ConfigurationError,
    load_application_settings,
    load_policy_settings,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _environment(**overrides: str) -> dict[str, str]:
    environment = {
        "DATABASE_URL": "postgresql+psycopg://user:password@db/voice",
        "CELERY_BROKER_URL": "redis://redis:6379/0",
        "S3_BUCKET": "test-bucket",
        "S3_REGION": "us-east-1",
    }
    environment.update(overrides)
    return environment


def test_bundled_policy_defaults_are_loaded() -> None:
    settings = load_application_settings(_environment())

    assert settings.policy.ingest.max_upload_bytes == 314_572_800
    assert settings.policy.ingest.max_concurrent_requests == 10


def test_external_policy_overrides_only_explicit_keys(tmp_path: Path) -> None:
    override = _write(
        tmp_path / "override.toml",
        "[ingest]\nmax_concurrent_requests = 3\n",
    )
    settings = load_application_settings(_environment(INGEST_CONFIG_FILE=str(override)))

    assert settings.policy.ingest.max_upload_bytes == 314_572_800
    assert settings.policy.ingest.max_concurrent_requests == 3


@pytest.mark.parametrize(
    "content",
    [
        "[ingest]\nunknown = 1\n",
        '[ingest]\nmax_upload_bytes = "large"\n',
        "[ingest]\nmax_concurrent_requests = 0\n",
    ],
)
def test_invalid_policy_prevents_startup(
    tmp_path: Path,
    content: str,
) -> None:
    default = _write(
        tmp_path / "default.toml",
        ("[ingest]\nmax_upload_bytes = 314572800\nmax_concurrent_requests = 10\n"),
    )
    override = _write(tmp_path / "override.toml", content)

    with pytest.raises(ConfigurationError, match="Policy configuration"):
        load_policy_settings(default_path=default, override_path=override)


def test_business_environment_variables_do_not_override_policy() -> None:
    settings = load_application_settings(
        _environment(
            INGEST_MAX_UPLOAD_BYTES="1",
            INGEST_MAX_CONCURRENT_REQUESTS="1",
        )
    )

    assert settings.policy.ingest.max_upload_bytes == 314_572_800
    assert settings.policy.ingest.max_concurrent_requests == 10


def test_aws_sdk_credentials_are_not_copied_into_service_settings() -> None:
    settings = load_application_settings(
        _environment(
            AWS_ACCESS_KEY_ID="test-access-key",
            AWS_SECRET_ACCESS_KEY="test-secret-key",
        )
    )

    serialized = repr(settings)
    assert "test-access-key" not in serialized
    assert "test-secret-key" not in serialized


def test_missing_required_environment_variable_prevents_startup() -> None:
    environment = _environment()
    del environment["DATABASE_URL"]

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        load_application_settings(environment)


def test_default_dotenv_is_loaded_without_overriding_process_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(
        tmp_path / ".env",
        "DATABASE_URL=postgresql+psycopg://dotenv@db/voice\n"
        "CELERY_BROKER_URL=redis://dotenv-redis:6379/0\n"
        "S3_BUCKET=dotenv-bucket\n"
        "S3_REGION=us-west-2\n"
        "AWS_ACCESS_KEY_ID=dotenv-access-key\n"
        "AWS_SECRET_ACCESS_KEY=dotenv-secret-key\n",
    )
    for name in (
        "DATABASE_URL",
        "CELERY_BROKER_URL",
        "S3_BUCKET",
        "S3_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("S3_REGION", "eu-central-1")
    monkeypatch.chdir(tmp_path)

    settings = load_application_settings()

    assert settings.environment.database_url == ("postgresql+psycopg://dotenv@db/voice")
    assert settings.environment.celery_broker_url == "redis://dotenv-redis:6379/0"
    assert settings.environment.s3_bucket == "dotenv-bucket"
    assert settings.environment.s3_region == "eu-central-1"
