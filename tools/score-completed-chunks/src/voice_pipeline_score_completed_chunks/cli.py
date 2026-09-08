from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from . import __version__
from .aggregation import build_summary
from .audio_tag_accuracy import (
    DEFAULT_MODEL,
    AudioTagEvaluator,
    AudioTagScoreEngine,
    summarize_audio_tag_rows,
)
from .config import Settings
from .dnsmos import DnsmosScorer
from .interaction_aggregation import build_interaction_summary
from .interaction_config import InteractionConfig
from .interaction_scoring import InteractionScoreEngine
from .nisqa import NisqaScorer
from .nonverbal import AstNonverbalDetector, DisabledNonverbalDetector
from .outputs import RunOutputs, atomic_json
from .repository import Repository
from .scoring import ScoreEngine
from .speaker_similarity import SpeakerSimilarityScorer
from .storage import ObjectStorage
from .vad import EnergyVad


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score separation, reconstruction, and expansion audio for completed "
            "chunks."
        )
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--chunk-id", action="append", type=UUID, default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-audio-tag-evaluation", action="store_true")
    parser.add_argument("--audio-tag-inline-only", action="store_true")
    parser.add_argument("--audio-tag-model", default=DEFAULT_MODEL)
    parser.add_argument("--audio-tag-workers", type=int, default=4)
    parser.add_argument("--audio-tag-timeout", type=float, default=120.0)
    interaction = parser.add_mutually_exclusive_group()
    interaction.add_argument("--interaction-coverage", action="store_true")
    interaction.add_argument("--skip-interaction-coverage", action="store_true")
    parser.add_argument("--skip-interaction-nonverbal", action="store_true")
    parser.add_argument("--interaction-bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--interaction-seed", type=int, default=20_260_828)
    return parser


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "onnxruntime",
        "requests",
        "torch",
        "torchaudio",
        "torchmetrics",
        "transformers",
    )
    return {name: importlib.metadata.version(name) for name in names}


def _fingerprint() -> str:
    policy = {
        "schema_version": 1,
        "active_audio": "transcript-intervals-merge-100ms-separator-v1",
        "nisqa": "nisqa-v2-general-7ec4cf937514-window50-duration-mean-v2",
        "dnsmos": "p835-regular-591184a9fcb2",
        "speaker_similarity": "wavlm-feb593a6c23c-window10-duration-pool-v1",
    }
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.audio_tag_workers <= 0:
        raise SystemExit("--audio-tag-workers must be positive")
    if args.audio_tag_timeout <= 0:
        raise SystemExit("--audio-tag-timeout must be positive")
    if args.interaction_bootstrap_samples <= 0:
        raise SystemExit("--interaction-bootstrap-samples must be positive")
    interaction_enabled = bool(args.interaction_coverage)
    settings = Settings.load(
        output_dir=args.output_dir,
        env_file=args.env_file,
        model_cache_dir=args.model_cache_dir,
    )
    if not args.skip_audio_tag_evaluation and not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required unless --skip-audio-tag-evaluation is used"
        )
    outputs = RunOutputs(settings.output_dir, resume=args.resume)
    started = datetime.now(UTC)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "tool_version": __version__,
        "status": "running",
        "started_at": started.isoformat(),
        "finished_at": None,
        "repository_git_commit": _git_commit(settings.repository_root),
        "query": {
            "status": "completed",
            "chunk_ids": [str(value) for value in args.chunk_id],
            "limit": args.limit,
        },
        "configuration": {
            "device": args.device,
            "resume": args.resume,
            "fail_fast": args.fail_fast,
            "s3_bucket": settings.s3_bucket,
            "s3_region": settings.s3_region,
            "s3_endpoint_configured": settings.s3_endpoint_url is not None,
            "database_url_recorded": False,
            "audio_tag_evaluation": not args.skip_audio_tag_evaluation,
            "audio_tag_inline_only": args.audio_tag_inline_only,
            "audio_tag_model": args.audio_tag_model,
            "audio_tag_workers": args.audio_tag_workers,
            "audio_tag_timeout_seconds": args.audio_tag_timeout,
            "interaction_coverage": interaction_enabled,
            "interaction_nonverbal": interaction_enabled
            and not args.skip_interaction_nonverbal,
            "interaction_bootstrap_samples": args.interaction_bootstrap_samples,
            "interaction_seed": args.interaction_seed,
        },
        "versions": _package_versions(),
        "model_fingerprint": _fingerprint(),
        "models": None,
        "counts": None,
    }
    atomic_json(settings.output_dir / "run-manifest.json", manifest)
    print(f"Output: {settings.output_dir}", flush=True)

    storage = ObjectStorage(
        bucket=settings.s3_bucket,
        region=settings.s3_region,
        endpoint_url=settings.s3_endpoint_url,
    )
    failures = 0
    chunks = 0
    speaker_attempts = 0
    audio_tag_attempts = 0
    interaction_attempts = 0
    try:
        print("Loading metric models...", flush=True)
        nisqa = NisqaScorer(settings.model_cache_dir)
        dnsmos = DnsmosScorer(settings.model_cache_dir)
        speaker = SpeakerSimilarityScorer(args.device, settings.model_cache_dir)
        engine = ScoreEngine(
            storage,
            nisqa,
            dnsmos,
            speaker,
            str(manifest["model_fingerprint"]),
        )
        audio_tag_engine = None
        audio_tag_evaluator = None
        if not args.skip_audio_tag_evaluation:
            assert settings.openrouter_api_key is not None
            audio_tag_evaluator = AudioTagEvaluator(
                api_key=settings.openrouter_api_key,
                model=args.audio_tag_model,
                timeout_seconds=args.audio_tag_timeout,
            )
            audio_tag_engine = AudioTagScoreEngine(
                storage,
                audio_tag_evaluator,
                workers=args.audio_tag_workers,
                inline_only=args.audio_tag_inline_only,
            )
        interaction_config = None
        interaction_engine = None
        if interaction_enabled:
            interaction_config = InteractionConfig(
                bootstrap_samples=args.interaction_bootstrap_samples,
                seed=args.interaction_seed,
            )
            vad = EnergyVad(interaction_config)
            nonverbal = (
                DisabledNonverbalDetector()
                if args.skip_interaction_nonverbal
                else AstNonverbalDetector(
                    device=args.device,
                    model_cache=settings.model_cache_dir,
                    threshold=interaction_config.nonverbal_threshold,
                )
            )
            interaction_engine = InteractionScoreEngine(
                storage=storage,
                config=interaction_config,
                vad=vad,
                nonverbal=nonverbal,
            )
        manifest["models"] = {
            "nisqa": nisqa.manifest(),
            "dnsmos": dnsmos.manifest(),
            "speaker_similarity": speaker.manifest(),
            "audio_tag_accuracy": (
                audio_tag_evaluator.manifest() if audio_tag_evaluator else None
            ),
            "interaction": (
                interaction_engine.manifest() if interaction_engine else None
            ),
        }
        atomic_json(settings.output_dir / "run-manifest.json", manifest)
        repository = Repository(settings.database_url)
        for chunk in repository.iter_completed(
            chunk_ids=tuple(args.chunk_id), limit=args.limit
        ):
            chunks += 1
            print(f"[{chunks}] {chunk.chunk_id}", flush=True)
            speaker_rows, group_rows, failure_rows = engine.score_chunk(
                chunk, existing=outputs.existing_speakers
            )
            for row in speaker_rows:
                key = (row["chunk_id"], row["group"], row["speaker_id"])
                if outputs.existing_speakers.get(key) is not row:
                    outputs.write_speaker(row)
                speaker_attempts += 1
            for row in group_rows:
                outputs.write_group(row)
            for row in failure_rows:
                outputs.write_failure(row)
            failures += len(failure_rows)
            chunk_audio_tag_rows: list[dict] = []
            if audio_tag_engine is not None:
                audio_tag_rows, audio_tag_failures = audio_tag_engine.score_chunk(
                    chunk, existing=outputs.existing_audio_tags
                )
                for row in audio_tag_rows:
                    chunk_audio_tag_rows.append(row)
                    key = (
                        row["chunk_id"],
                        row["group"],
                        row["transcript_index"],
                    )
                    if outputs.existing_audio_tags.get(key) is not row:
                        outputs.write_audio_tag(row)
                    else:
                        outputs.audio_tag_rows.append(row)
                    audio_tag_attempts += 1
                for row in audio_tag_failures:
                    outputs.write_failure(row)
                failures += len(audio_tag_failures)
                failure_rows.extend(audio_tag_failures)
            if interaction_engine is not None:
                (
                    interaction_events,
                    interaction_scores,
                    interaction_declared,
                    interaction_failures,
                ) = interaction_engine.score_chunk(
                    chunk,
                    audio_tag_rows=chunk_audio_tag_rows,
                )
                for row in interaction_scores:
                    outputs.write_interaction_score(row)
                    interaction_attempts += 1
                for row in interaction_events:
                    outputs.write_interaction_event(row)
                for row in interaction_declared:
                    outputs.write_interaction_declared(row)
                for row in interaction_failures:
                    outputs.write_failure(row)
                failures += len(interaction_failures)
                failure_rows.extend(interaction_failures)
            if failure_rows and args.fail_fast:
                raise RuntimeError("scoring stopped by --fail-fast")
        summary = build_summary(outputs.group_rows)
        audio_tag_summary = summarize_audio_tag_rows(outputs.audio_tag_rows)
        summary["audio_tag_accuracy"] = audio_tag_summary
        outputs.write_summary(summary)
        outputs.write_audio_tag_summary(audio_tag_summary)
        if interaction_config is not None:
            interaction_summary, interaction_bootstrap, interaction_paired = (
                build_interaction_summary(
                    outputs.interaction_score_rows,
                    outputs.interaction_declared_rows,
                    config=interaction_config,
                )
            )
            outputs.write_interaction_summary(
                interaction_summary,
                interaction_bootstrap,
                interaction_paired,
            )
        finished = datetime.now(UTC)
        manifest["status"] = "complete" if failures == 0 else "partial"
        manifest["finished_at"] = finished.isoformat()
        manifest["counts"] = {
            "chunks": chunks,
            "speaker_attempts": speaker_attempts,
            "group_attempts": len(outputs.group_rows),
            "audio_tag_utterance_attempts": audio_tag_attempts,
            "failure_records": failures,
            "interaction_group_attempts": interaction_attempts,
        }
        atomic_json(settings.output_dir / "run-manifest.json", manifest)
        print(
            f"Finished: {chunks} chunks, {failures} failure records. "
            f"Summary: {settings.output_dir / 'summary.json'}",
            flush=True,
        )
        return 0 if failures == 0 else 2
    except Exception:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["counts"] = {
            "chunks": chunks,
            "speaker_attempts": speaker_attempts,
            "group_attempts": len(outputs.group_rows),
            "audio_tag_utterance_attempts": audio_tag_attempts,
            "failure_records": failures,
            "interaction_group_attempts": interaction_attempts,
        }
        atomic_json(settings.output_dir / "run-manifest.json", manifest)
        raise
    finally:
        outputs.close()
        storage.close()


def main() -> None:
    try:
        status = run(_parser().parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"score-completed-chunks: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
