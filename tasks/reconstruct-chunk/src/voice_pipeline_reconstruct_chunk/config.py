from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)


class ConfigurationError(RuntimeError):
    """Raised when worker configuration is missing or invalid."""


class AudioTagsPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    model: str
    reasoning_effort: Literal["none"]
    max_tokens: int = Field(gt=0, le=4096)
    timeout_seconds: int = Field(gt=0, le=600)
    max_attempts: int = Field(gt=0, le=3)
    retry_backoff_seconds: float = Field(ge=0, le=60)
    require_parameters: bool
    allow_fallbacks: bool

    @field_validator("model")
    @classmethod
    def model_id(cls, value: str) -> str:
        if not value or value != value.strip() or "/" not in value:
            raise ValueError("model must be a canonical provider model ID")
        return value


class TtsPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    model: str
    timeout_seconds: int = Field(gt=0, le=600)
    max_attempts: int = Field(gt=0, le=5)
    retry_backoff_seconds: float = Field(ge=0, le=60)
    temperature: float = Field(ge=0, le=1)
    top_p: float = Field(gt=0, le=1)
    chunk_length: int = Field(ge=100, le=300)
    min_chunk_length: int = Field(ge=0, le=100)
    max_new_tokens: int = Field(gt=0, le=8192)
    repetition_penalty: float = Field(gt=0, le=4)
    sample_rate_hz: Literal[44100]
    latency: Literal["normal", "balanced", "low"]
    normalize_text: bool
    normalize_loudness: bool

    @field_validator("model")
    @classmethod
    def canonical_model(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("model must be a non-empty canonical model name")
        return value


class ForcedAlignmentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    repo_id: Literal["Qwen/Qwen3-ForcedAligner-0.6B"]
    revision: str
    device: Literal["cuda:0", "cpu"]
    dtype: Literal["bfloat16", "float32"]

    @field_validator("revision")
    @classmethod
    def pinned_revision(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("forced aligner revision must be a pinned commit")
        return value


class AudioPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    input_sample_rate_hz: Literal[16000]
    reference_silence_ms: Literal[1000]
    output_sample_rate_hz: Literal[44100]


class ReferencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    fallback_maximum_duration_ms: int = Field(gt=0, le=60000)
    fallback_edge_trim_ms: int = Field(ge=0, le=5000)


class TaskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    workspace_prefix: str
    error_max_length: int = Field(gt=0, le=4096)

    @field_validator("workspace_prefix")
    @classmethod
    def safe_prefix(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            raise ValueError("workspace_prefix must be safe")
        return value


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    config_version: Literal["source-reconstruction-v1"]
    audio_tags: AudioTagsPolicy
    tts: TtsPolicy
    forced_alignment: ForcedAlignmentPolicy
    audio: AudioPolicy
    reference: ReferencePolicy
    task: TaskPolicy


class Environment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    database_url: str
    celery_broker_url: str
    s3_bucket: str
    s3_region: str
    openrouter_api_key: SecretStr
    s3_endpoint_url: str | None = None

    @field_validator("database_url", "celery_broker_url", "s3_bucket", "s3_region")
    @classmethod
    def nonempty(cls, value: str) -> str:
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
    required = (
        "DATABASE_URL",
        "CELERY_BROKER_URL",
        "S3_BUCKET",
        "S3_REGION",
        "OPENROUTER_API_KEY",
    )
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}."
        )
    path = Path(environment.get("RECONSTRUCT_CHUNK_CONFIG_FILE", str(DEFAULT_POLICY)))
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
                openrouter_api_key=environment["OPENROUTER_API_KEY"],
                s3_endpoint_url=environment.get("S3_ENDPOINT_URL"),
            ),
        )
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigurationError("Task configuration is invalid.") from exc
