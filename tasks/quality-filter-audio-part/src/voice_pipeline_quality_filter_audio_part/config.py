"""Validated policy and deployment settings."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigurationError(RuntimeError):
    """Raised when worker configuration is absent or invalid."""


class QualityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    min_snr_db: float = Field(ge=0.0)
    music_probability_threshold: float = Field(ge=0.0, le=1.0)
    min_music_interval_ms: int = Field(ge=0)
    music_gap_fill_ms: int = Field(ge=0)
    max_music_overlap_ratio: float = Field(ge=0.0, le=1.0)
    max_absorbable_bad_group_ms: int = Field(gt=0)
    min_good_region_ms: int = Field(gt=0)


class PlannerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    min_planning_window_ms: int = Field(gt=0)
    max_planning_window_ms: int = Field(gt=0)
    min_speaker_turn_ms: int = Field(gt=0)
    min_speaker_total_ms: int = Field(gt=0)
    backchannel_threshold_ms: int = Field(gt=0)
    max_monologue_ms: int = Field(gt=0)
    max_inner_iterations: int = Field(gt=0, le=10000)

    @model_validator(mode="after")
    def validate_bounds(self) -> PlannerPolicy:
        if self.max_planning_window_ms < self.min_planning_window_ms:
            raise ValueError("planning window bounds are inconsistent")
        if self.min_speaker_turn_ms > self.min_speaker_total_ms:
            raise ValueError("speaker thresholds are inconsistent")
        return self


class MusicPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    model_name: str
    model_filename: str
    mean_filename: str
    std_filename: str
    model_sha256: str
    mean_sha256: str
    std_sha256: str
    sample_rate: int = Field(gt=0)
    fft_size: int = Field(gt=0)
    hop_length: int = Field(gt=0)
    mel_bins: int = Field(gt=0)
    min_frequency_hz: float = Field(gt=0)
    max_frequency_hz: float = Field(gt=0)

    @field_validator("model_name", "model_filename", "mean_filename", "std_filename")
    @classmethod
    def require_safe_name(cls, value: str) -> str:
        if not value.strip() or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("must be a non-empty safe name")
        return value

    @field_validator("model_sha256", "mean_sha256", "std_sha256")
    @classmethod
    def validate_optional_checksum(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized and (len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized)):
            raise ValueError("must be an empty value or a SHA-256 hex digest")
        return normalized

    @model_validator(mode="after")
    def validate_frequency_range(self) -> MusicPolicy:
        if self.max_frequency_hz <= self.min_frequency_hz:
            raise ValueError("frequency bounds are inconsistent")
        return self


class TaskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    error_max_length: int = Field(gt=0, le=4096)
    workspace_prefix: str
    max_diarization_bytes: int = Field(gt=0)

    @field_validator("workspace_prefix")
    @classmethod
    def require_safe_prefix(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("must be a safe directory prefix")
        return value


class PolicySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_version: int
    quality: QualityPolicy
    planner: PlannerPolicy
    music: MusicPolicy
    task: TaskPolicy

    @field_validator("config_version")
    @classmethod
    def require_version_one(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported config version")
        return value


class EnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_url: str
    celery_broker_url: str
    s3_bucket: str
    s3_region: str
    s3_endpoint_url: str | None = None
    music_model_cache_dir: Path

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
DEFAULT_MUSIC_MODEL_CACHE_DIR = Path(__file__).with_name("music_artifacts")
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
    missing = sorted(name for name in _REQUIRED_ENVIRONMENT_VARIABLES if not source.get(name))
    if missing:
        raise ConfigurationError(f"Missing required environment variables: {', '.join(missing)}.")
    document = _read_toml(default_policy_path)
    selected_override = override_policy_path
    if selected_override is None and _optional(source, "QUALITY_FILTER_CONFIG_FILE"):
        selected_override = Path(source["QUALITY_FILTER_CONFIG_FILE"])
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
                music_model_cache_dir=Path(
                    _optional(source, "MUSIC_MODEL_CACHE_DIR")
                    or DEFAULT_MUSIC_MODEL_CACHE_DIR
                ),
            ),
        )
    except ValidationError as exc:
        raise ConfigurationError("Task configuration is invalid.") from exc
