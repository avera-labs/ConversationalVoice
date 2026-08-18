"""Validated policy and environment configuration."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)


class ConfigurationError(RuntimeError):
    """Raised when configuration cannot be loaded safely."""


class DiarizationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: str
    device: Literal["auto", "cpu", "cuda"]

    @field_validator("model")
    @classmethod
    def require_hugging_face_model_name(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("must not be empty")
        try:
            validate_repo_id(model)
        except HFValidationError as exc:
            raise ValueError("must be a valid Hugging Face model name") from exc
        return model


class SpeakerReferencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    min_segment_ms: int = Field(gt=0)
    edge_trim_ms: int = Field(ge=0)
    min_speaker_effective_ms: int = Field(gt=0)
    max_speaker_effective_ms: int = Field(gt=0)
    inter_segment_silence_ms: int = Field(ge=0)

    @field_validator("max_speaker_effective_ms")
    @classmethod
    def require_valid_effective_duration_limit(cls, value: int, info: Any) -> int:
        minimum = info.data.get("min_speaker_effective_ms")
        if isinstance(minimum, int) and value < minimum:
            raise ValueError("must not be less than min_speaker_effective_ms")
        return value


class TaskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    error_max_length: int = Field(gt=0, le=4096)
    workspace_prefix: str

    @field_validator("workspace_prefix")
    @classmethod
    def require_safe_prefix(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("must be a safe directory prefix")
        return value


class PolicySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    diarization: DiarizationPolicy
    speaker_reference: SpeakerReferencePolicy
    task: TaskPolicy


class EnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_url: str
    celery_broker_url: str
    s3_bucket: str
    s3_region: str
    s3_endpoint_url: str | None = None

    @field_validator("database_url", "celery_broker_url", "s3_bucket", "s3_region")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class Settings(BaseModel):
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
        raise ConfigurationError("Unable to load policy configuration.") from exc
    if not isinstance(document, dict):
        raise ConfigurationError("Policy configuration is invalid.")
    return document


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _optional(source: Mapping[str, str], name: str) -> str | None:
    value = source.get(name)
    return value if value and value.strip() else None


def load_settings(
    environment: Mapping[str, str] | None = None,
    *,
    default_policy_path: Path = DEFAULT_POLICY_PATH,
    override_policy_path: Path | None = None,
) -> Settings:
    if environment is None:
        load_dotenv(DEFAULT_ENV_FILE_PATH, override=False)
        source: Mapping[str, str] = os.environ
    else:
        source = environment
    missing = sorted(
        name for name in _REQUIRED_ENVIRONMENT_VARIABLES if not source.get(name)
    )
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}."
        )

    document = _read_toml(default_policy_path)
    selected_override = override_policy_path
    if selected_override is None and _optional(source, "DIARIZATION_CONFIG_FILE"):
        selected_override = Path(source["DIARIZATION_CONFIG_FILE"])
    if selected_override is not None:
        document = _merge(document, _read_toml(selected_override))
    try:
        return Settings(
            policy=PolicySettings.model_validate(document),
            environment=EnvironmentSettings(
                database_url=source["DATABASE_URL"],
                celery_broker_url=source["CELERY_BROKER_URL"],
                s3_bucket=source["S3_BUCKET"],
                s3_region=source["S3_REGION"],
                s3_endpoint_url=_optional(source, "S3_ENDPOINT_URL"),
            ),
        )
    except ValidationError as exc:
        raise ConfigurationError("Task configuration is invalid.") from exc
