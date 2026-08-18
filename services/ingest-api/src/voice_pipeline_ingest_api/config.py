from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigurationError(RuntimeError):
    """Raised when service configuration cannot be loaded safely."""


class IngestPolicy(BaseModel):
    """Non-sensitive ingest policy loaded from TOML."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_upload_bytes: int = Field(gt=0)
    max_concurrent_requests: int = Field(gt=0)


class PolicySettings(BaseModel):
    """Validated policy configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ingest: IngestPolicy


class EnvironmentSettings(BaseModel):
    """Deployment-specific settings loaded only from environment variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    database_url: str
    celery_broker_url: str
    s3_bucket: str
    s3_region: str
    s3_endpoint_url: str | None = None
    ingest_config_file: Path | None = None

    @field_validator(
        "database_url",
        "celery_broker_url",
        "s3_bucket",
        "s3_region",
    )
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class ApplicationSettings(BaseModel):
    """Complete immutable service configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: PolicySettings
    environment: EnvironmentSettings


DEFAULT_POLICY_PATH = Path(__file__).with_name("resources") / "default.toml"
DEFAULT_ENV_FILE_PATH = Path(".env")

_REQUIRED_ENVIRONMENT_VARIABLES = (
    "DATABASE_URL",
    "CELERY_BROKER_URL",
    "S3_BUCKET",
    "S3_REGION",
)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            document = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(
            f"Unable to load policy configuration from {path}."
        ) from exc

    if not isinstance(document, dict):
        raise ConfigurationError(f"Policy configuration at {path} is invalid.")
    return document


def _merge_mappings(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def _optional_environment_value(
    environment: Mapping[str, str],
    name: str,
) -> str | None:
    value = environment.get(name)
    if value is None or not value.strip():
        return None
    return value


def load_environment_settings(
    environment: Mapping[str, str] | None = None,
) -> EnvironmentSettings:
    source = os.environ if environment is None else environment
    missing = [name for name in _REQUIRED_ENVIRONMENT_VARIABLES if not source.get(name)]
    if missing:
        names = ", ".join(sorted(missing))
        raise ConfigurationError(f"Missing required environment variables: {names}.")

    try:
        return EnvironmentSettings(
            database_url=source["DATABASE_URL"],
            celery_broker_url=source["CELERY_BROKER_URL"],
            s3_bucket=source["S3_BUCKET"],
            s3_region=source["S3_REGION"],
            s3_endpoint_url=_optional_environment_value(source, "S3_ENDPOINT_URL"),
            ingest_config_file=_optional_environment_value(
                source, "INGEST_CONFIG_FILE"
            ),
        )
    except ValidationError as exc:
        raise ConfigurationError("Environment configuration is invalid.") from exc


def load_policy_settings(
    *,
    default_path: Path = DEFAULT_POLICY_PATH,
    override_path: Path | None = None,
) -> PolicySettings:
    merged = _read_toml(default_path)
    if override_path is not None:
        merged = _merge_mappings(merged, _read_toml(override_path))

    try:
        return PolicySettings.model_validate(merged)
    except ValidationError as exc:
        raise ConfigurationError("Policy configuration is invalid.") from exc


def load_application_settings(
    environment: Mapping[str, str] | None = None,
    *,
    default_policy_path: Path = DEFAULT_POLICY_PATH,
) -> ApplicationSettings:
    if environment is None:
        load_dotenv(dotenv_path=DEFAULT_ENV_FILE_PATH, override=False)
    environment_settings = load_environment_settings(environment)
    policy_settings = load_policy_settings(
        default_path=default_policy_path,
        override_path=environment_settings.ingest_config_file,
    )
    return ApplicationSettings(
        policy=policy_settings,
        environment=environment_settings,
    )
