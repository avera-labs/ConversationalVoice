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


class OpenRouterPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    model: str
    max_tokens: int = Field(gt=0, le=65536)
    timeout_seconds: int = Field(gt=0, le=600)
    max_attempts: int = Field(gt=0, le=5)
    retry_backoff_seconds: float = Field(ge=0, le=60)
    require_parameters: bool
    allow_fallbacks: bool

    @field_validator("model")
    @classmethod
    def canonical_model(cls, value: str) -> str:
        if not value or value != value.strip() or "/" not in value:
            raise ValueError("model must be a canonical OpenRouter model ID")
        return value


class FishAudioPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    model: Literal["fish-audio/s2.1-pro"]
    transcription_model: Literal["fish-audio/transcribe-1"]
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


class DialoguePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    target_duration_seconds: int = Field(ge=10, le=1800)
    words_per_minute: int = Field(ge=80, le=240)
    min_utterances: int = Field(ge=2, le=100)
    max_utterances: int = Field(ge=2, le=200)

    @field_validator("max_utterances")
    @classmethod
    def valid_range(cls, value: int, info) -> int:
        minimum = info.data.get("min_utterances")
        if isinstance(minimum, int) and value < minimum:
            raise ValueError("max_utterances must not be below min_utterances")
        return value

    @property
    def target_duration_ms(self) -> int:
        return self.target_duration_seconds * 1000

    @property
    def target_words(self) -> int:
        return round(self.target_duration_seconds * self.words_per_minute / 60)


class TimelinePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    turn_gap_ms: int = Field(ge=0, le=5000)
    overlap_ms: int = Field(gt=0, le=5000)
    same_speaker_gap_ms: int = Field(ge=0, le=1000)


class TaskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    workspace_prefix: str
    error_max_length: int = Field(gt=0, le=4096)

    @field_validator("workspace_prefix")
    @classmethod
    def safe_prefix(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            raise ValueError("workspace_prefix must be a safe name prefix")
        return value


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    config_version: Literal["dialogue-extension-v1"]
    openrouter: OpenRouterPolicy
    fish_audio: FishAudioPolicy
    dialogue: DialoguePolicy
    timeline: TimelinePolicy
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

    @field_validator("openrouter_api_key")
    @classmethod
    def nonempty_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
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
    path = Path(environment.get("EXTEND_CHUNK_CONFIG_FILE", str(DEFAULT_POLICY)))
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
