from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv


def find_repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (
            candidate / "schema/schema.sql"
        ).is_file():
            return candidate
    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if (candidate / ".git").exists() and (
            candidate / "schema/schema.sql"
        ).is_file():
            return candidate
    raise RuntimeError("repository root could not be found")


def default_output_dir(repository_root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    return repository_root / "outputs" / "chunk-quality" / stamp


def default_model_cache() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "conversational-voice" / "metrics"


@dataclass(frozen=True, slots=True)
class Settings:
    repository_root: Path
    database_url: str
    s3_bucket: str
    s3_region: str
    s3_endpoint_url: str | None
    openrouter_api_key: str | None
    output_dir: Path
    model_cache_dir: Path

    @classmethod
    def load(
        cls,
        *,
        output_dir: Path | None = None,
        env_file: Path | None = None,
        model_cache_dir: Path | None = None,
    ) -> "Settings":
        root = find_repository_root()
        load_dotenv(env_file or root / ".env", override=False)
        database_url = os.environ.get("DATABASE_URL", "").strip()
        bucket = os.environ.get("S3_BUCKET", "").strip()
        if not database_url:
            raise RuntimeError("DATABASE_URL is required")
        if not bucket:
            raise RuntimeError("S3_BUCKET is required")
        endpoint = os.environ.get("S3_ENDPOINT_URL", "").strip() or None
        return cls(
            repository_root=root,
            database_url=database_url,
            s3_bucket=bucket,
            s3_region=os.environ.get("S3_REGION", "us-east-1").strip() or "us-east-1",
            s3_endpoint_url=endpoint,
            openrouter_api_key=(
                os.environ.get("OPENROUTER_API_KEY", "").strip() or None
            ),
            output_dir=(output_dir or default_output_dir(root)).resolve(),
            model_cache_dir=(model_cache_dir or default_model_cache()).resolve(),
        )
