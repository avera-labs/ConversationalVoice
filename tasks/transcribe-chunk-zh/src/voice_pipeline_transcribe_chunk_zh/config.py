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
    """Raised when worker configuration is missing or invalid."""


class ModelPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    repo_id: Literal[
        "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    ]
    revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    device: Literal["auto", "cpu", "cuda"]


class PunctuationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    repo_id: Literal["iic/punc_ct-transformer_cn-en-common-vocab471067-large"]
    revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    max_chars: int = Field(gt=0, le=4096)


class SlicePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    merge_gap_ms: int = Field(gt=0)
    pad_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def padding_does_not_overlap(self):
        if 2 * self.pad_ms > self.merge_gap_ms:
            raise ValueError("twice pad_ms must not exceed merge_gap_ms")
        return self


class UtterancePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    huge_gap_ms: int = Field(gt=0)
    clause_min_ms: int = Field(gt=0)
    medium_min_ms: int = Field(gt=0)
    medium_gap_ms: int = Field(gt=0)
    emergency_min_ms: int = Field(gt=0)
    emergency_gap_ms: int = Field(gt=0)
    character_max_duration_ms: int = Field(gt=0)
    character_capped_duration_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def duration_bounds(self):
        if self.character_capped_duration_ms >= self.character_max_duration_ms:
            raise ValueError("capped character duration must be below the maximum")
        return self


class TaskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    workspace_prefix: str
    error_max_length: int = Field(gt=0, le=4096)

    @field_validator("workspace_prefix")
    @classmethod
    def safe_prefix(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("workspace_prefix must be a safe name prefix")
        return value


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    config_version: Literal["paraformer-zh-v1"]
    model: ModelPolicy
    punctuation: PunctuationPolicy
    slices: SlicePolicy
    utterance: UtterancePolicy
    task: TaskPolicy


class Environment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    database_url: str
    celery_broker_url: str
    s3_bucket: str
    s3_region: str
    s3_endpoint_url: str | None = None
    paraformer_model_dir: str | None = None
    punctuation_model_dir: str | None = None

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
    required = ("DATABASE_URL", "CELERY_BROKER_URL", "S3_BUCKET", "S3_REGION")
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}."
        )
    path = Path(environment.get("PARAFORMER_ZH_CONFIG_FILE", str(DEFAULT_POLICY)))
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
                paraformer_model_dir=environment.get("PARAFORMER_MODEL_DIR"),
                punctuation_model_dir=environment.get("CTPUNC_MODEL_DIR"),
            ),
        )
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigurationError("Task configuration is invalid.") from exc
