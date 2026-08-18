from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ConfigurationError(RuntimeError):
    pass


class ModelPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    repo_id: str
    revision: str
    inference_steps: int = Field(gt=0)
    device: Literal["auto", "cpu", "cuda"]


class AlignmentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    repo_id: str
    revision: str
    similarity_min: float = Field(ge=-1, le=1)
    margin_min: float = Field(ge=0, le=2)
    voice_rms_min: float = Field(gt=0)


class WindowPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    initial_ms: int = Field(gt=0)
    maximum_ms: int = Field(gt=0)
    extension_ms: int = Field(gt=0)
    overlap_ms: int = Field(gt=0)
    crossfade_ms: int = Field(gt=0)
    speaker_once_ms: int = Field(gt=0)
    speaker_total_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def bounds(self):
        if (
            not (self.overlap_ms < self.initial_ms <= self.maximum_ms)
            or self.crossfade_ms > self.overlap_ms
        ):
            raise ValueError("window bounds are inconsistent")
        return self


class AuditPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    pure_min_ms: int = Field(gt=0)
    trim_ms: int = Field(ge=0)
    min_rms_ratio: float = Field(gt=1)
    require_both_speakers: bool


class OutputPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    sample_rate_hz: int = Field(gt=0)
    peak: float = Field(gt=0, le=1)
    silence_peak_max: float = Field(ge=0)


class TaskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    workspace_prefix: str
    error_max_length: int = Field(gt=0, le=4096)
    max_diarization_bytes: int = Field(gt=0)

    @field_validator("workspace_prefix")
    @classmethod
    def require_safe_prefix(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("must be a safe directory prefix")
        return value


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    config_version: Literal["sidon-v1"]
    model: ModelPolicy
    alignment: AlignmentPolicy
    window: WindowPolicy
    audit: AuditPolicy
    output: OutputPolicy
    task: TaskPolicy


class Environment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    database_url: str
    celery_broker_url: str
    s3_bucket: str
    s3_region: str
    s3_endpoint_url: str | None = None
    hf_token: str | None = None

    @field_validator("database_url", "celery_broker_url", "s3_bucket", "s3_region")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy: Policy
    environment: Environment


DEFAULT_POLICY = Path(__file__).with_name("resources") / "default.toml"


def load_settings(environment: Mapping[str, str] | None = None) -> Settings:
    if environment is None:
        load_dotenv(Path(".env"), override=False)
        environment = os.environ
    required = ("DATABASE_URL", "CELERY_BROKER_URL", "S3_BUCKET", "S3_REGION")
    missing = [key for key in required if not environment.get(key)]
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}."
        )
    path = Path(environment.get("SIDON_CONFIG_FILE", str(DEFAULT_POLICY)))
    try:
        with path.open("rb") as stream:
            policy = Policy.model_validate(tomllib.load(stream))
        return Settings(
            policy=policy,
            environment=Environment(
                database_url=environment["DATABASE_URL"],
                celery_broker_url=environment["CELERY_BROKER_URL"],
                s3_bucket=environment["S3_BUCKET"],
                s3_region=environment["S3_REGION"],
                s3_endpoint_url=environment.get("S3_ENDPOINT_URL"),
                hf_token=environment.get("HF_TOKEN"),
            ),
        )
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigurationError("Task configuration is invalid.") from exc
