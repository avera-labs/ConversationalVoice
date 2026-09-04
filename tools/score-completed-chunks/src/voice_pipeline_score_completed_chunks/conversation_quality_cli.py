from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from . import __version__
from .config import Settings
from .conversation_quality import (
    DEFAULT_MODEL,
    ConversationQualityEvaluator,
    ConversationQualityScoreEngine,
    summarize_conversation_quality_rows,
)
from .errors import error_code
from .outputs import JsonlAppender, atomic_json, read_jsonl
from .repository import Repository
from .storage import ObjectStorage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use Gemini audio input to score Expansion coherence with Reconstruction "
            "and Expansion two-person dialogue naturalness."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--chunk-id", action="append", type=UUID, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if len(set(args.chunk_id)) != len(args.chunk_id):
        raise SystemExit("--chunk-id values must be unique")
    settings = Settings.load(output_dir=args.output_dir, env_file=args.env_file)
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "conversation-quality-scores.jsonl"
    failure_path = output_dir / "failures.jsonl"
    if not args.resume and (score_path.exists() or failure_path.exists()):
        raise RuntimeError("output directory already contains a run; use --resume")
    existing_rows = read_jsonl(score_path) if args.resume else []
    existing = {row.get("chunk_id"): row for row in existing_rows}
    if args.resume:
        score_path.write_text("", encoding="utf-8")
        failure_path.write_text("", encoding="utf-8")

    started = datetime.now(UTC)
    evaluator = ConversationQualityEvaluator(
        api_key=settings.openrouter_api_key,
        model=args.model,
        timeout_seconds=args.timeout,
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "tool_version": __version__,
        "status": "running",
        "started_at": started.isoformat(),
        "finished_at": None,
        "query": {
            "status": "completed",
            "chunk_ids": [str(chunk_id) for chunk_id in args.chunk_id],
        },
        "evaluator": evaluator.manifest(),
        "request_protocol": {
            "requests_per_chunk": 1,
            "submitted_audio": ["reconstruction", "expansion"],
            "submitted_transcripts": False,
            "scores": ["content_coherence", "dialogue_naturalness"],
            "score_range": [1.0, 5.0],
            "score_increment": 0.1,
        },
        "counts": None,
    }
    atomic_json(output_dir / "run-manifest.json", manifest)
    print(f"Output: {output_dir}", flush=True)

    storage = ObjectStorage(
        bucket=settings.s3_bucket,
        region=settings.s3_region,
        endpoint_url=settings.s3_endpoint_url,
    )
    engine = ConversationQualityScoreEngine(storage, evaluator)
    repository = Repository(settings.database_url)
    score_output = JsonlAppender(score_path)
    failure_output = JsonlAppender(failure_path)
    rows: list[dict] = []
    found: set[UUID] = set()
    request_count = 0
    reused_count = 0
    try:
        for chunk in repository.iter_completed(chunk_ids=tuple(args.chunk_id)):
            found.add(chunk.chunk_id)
            print(f"Scoring {chunk.chunk_id}...", flush=True)
            try:
                row, reused = engine.score_chunk(
                    chunk, existing=existing.get(str(chunk.chunk_id))
                )
                score_output.write(row)
                rows.append(row)
                reused_count += int(reused)
                request_count += int(not reused)
                print(
                    f"  coherence={row['content_coherence']['score']} "
                    f"naturalness={row['dialogue_naturalness']['score']}"
                    + (" (reused)" if reused else ""),
                    flush=True,
                )
            except Exception as exc:
                failure_output.write(
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "scope": "conversation-quality",
                        "error_code": error_code(exc),
                    }
                )
                print(f"  failed: {error_code(exc)}", flush=True)
        for missing in args.chunk_id:
            if missing not in found:
                failure_output.write(
                    {
                        "chunk_id": str(missing),
                        "scope": "conversation-quality",
                        "error_code": "completed_chunk_not_found",
                    }
                )
    finally:
        score_output.close()
        failure_output.close()
        storage.close()

    summary = summarize_conversation_quality_rows(
        rows, requested_count=len(args.chunk_id)
    )
    atomic_json(output_dir / "conversation-quality-summary.json", summary)
    manifest["status"] = (
        "completed"
        if summary["failed_chunk_count"] == 0
        else "completed_with_failures"
    )
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["counts"] = {
        "requested_chunks": len(args.chunk_id),
        "successful_chunks": summary["successful_chunk_count"],
        "failed_chunks": summary["failed_chunk_count"],
        "provider_requests": request_count,
        "reused_rows": reused_count,
    }
    atomic_json(output_dir / "run-manifest.json", manifest)
    return 0 if summary["failed_chunk_count"] == 0 else 1


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))
