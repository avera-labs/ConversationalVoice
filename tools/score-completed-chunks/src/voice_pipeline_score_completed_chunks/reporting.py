from __future__ import annotations

import csv
import io
import math
from datetime import UTC, datetime

from .audio_tag_accuracy import summarize_audio_tag_rows
from .interaction_aggregation import jensen_shannon_distance
from .interaction_events import TRANSITION_CATEGORIES
from .outputs import canonical_json


def metric(
    value,
    *,
    unit: str,
    direction: str,
    support: object = None,
    status: str | None = None,
    reason: str | None = None,
    reference: str | None = None,
) -> dict[str, object]:
    valid = (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
    normalized_support = (
        support if isinstance(support, dict) or support is None else {"count": support}
    )
    return {
        "value": float(value) if valid else None,
        "unit": unit,
        "direction": direction,
        "status": status or ("ok" if valid else "not_computable"),
        "support": normalized_support,
        "reason": (reason or "metric was not produced") if not valid else None,
        "reference": reference,
    }


def _shared(group: dict | None, asr: dict | None, language: str) -> dict:
    group = group or {}
    asr = asr or {}
    error_name = "cer" if language.split("-", 1)[0].lower() == "zh" else "wer"
    asr_error = asr.get("error") if isinstance(asr.get("error"), dict) else {}
    return {
        error_name: metric(
            asr_error.get("value"),
            unit="ratio",
            direction="lower",
            support=asr_error.get("reference_unit_count"),
            reason=asr.get("error_code") or "ASR was not run",
        ),
        "nisqa_mos": metric(group.get("nisqa_mos"), unit="mos", direction="higher"),
        "dnsmos_ovrl": metric(group.get("dnsmos_ovrl"), unit="mos", direction="higher"),
        "same_speaker_similarity": metric(
            group.get("speaker_similarity"),
            unit="cosine",
            direction="higher",
            support=group.get("valid_speaker_count"),
        ),
        "speaker_discrimination_margin": metric(
            group.get("same_speaker_margin"),
            unit="cosine_difference",
            direction="higher",
            support=group.get("valid_speaker_count"),
        ),
    }


def _reconstruction(row: dict | None) -> dict:
    row = row or {}
    return {
        "duration_ratio": metric(
            row.get("duration_ratio"),
            unit="ratio",
            direction="target",
        ),
        "duration_log_error": metric(
            row.get("duration_log_error"),
            unit="absolute_log_ratio",
            direction="lower",
        ),
        "turn_event_f1": metric(
            row.get("turn_event_f1"),
            unit="ratio",
            direction="higher",
            support={
                "tp": row.get("turn_event_tp"),
                "fp": row.get("turn_event_fp"),
                "fn": row.get("turn_event_fn"),
            },
        ),
        "overlap_event_f1": metric(
            row.get("overlap_event_f1"),
            unit="ratio",
            direction="higher",
            support={
                "tp": row.get("overlap_event_tp"),
                "fp": row.get("overlap_event_fp"),
                "fn": row.get("overlap_event_fn"),
            },
        ),
        "backchannel_event_f1": metric(
            row.get("backchannel_event_f1"),
            unit="ratio",
            direction="higher",
            support={
                "tp": row.get("backchannel_event_tp"),
                "fp": row.get("backchannel_event_fp"),
                "fn": row.get("backchannel_event_fn"),
            },
        ),
    }


def _expansion(row: dict | None, reference: dict | None) -> dict:
    row = row or {}
    reference = reference or {}
    expansion_counts = row.get("transition_counts", {})
    reference_counts = reference.get("transition_counts", {})
    js_distance = (
        jensen_shannon_distance(
            [float(expansion_counts.get(name, 0)) for name in TRANSITION_CATEGORIES],
            [float(reference_counts.get(name, 0)) for name in TRANSITION_CATEGORIES],
        )
        if isinstance(expansion_counts, dict) and isinstance(reference_counts, dict)
        else None
    )
    paired_reference = "paired reconstruction; single-chunk diagnostic only"
    return {
        "expansion_factor": metric(
            row.get("expansion_factor"), unit="ratio", direction="descriptive"
        ),
        "turn_rate": metric(
            row.get("turn_rate_per_minute"),
            unit="events/min",
            direction="reference",
            reference=paired_reference,
        ),
        "backchannel_rate": metric(
            row.get("backchannel_rate_per_minute"),
            unit="events/min",
            direction="reference",
            reference=paired_reference,
        ),
        "interruption_rate": metric(
            row.get("interruption_rate_per_minute"),
            unit="events/min",
            direction="reference",
            reference=paired_reference,
        ),
        "overlap_event_rate": metric(
            row.get("overlap_event_rate_per_minute"),
            unit="events/min",
            direction="reference",
            reference=paired_reference,
        ),
        "transition_js_distance": metric(
            js_distance,
            unit="distance",
            direction="lower",
            reference=paired_reference,
            support=row.get("eligible_transition_count"),
        ),
        "effective_interaction_category_coverage": metric(
            row.get("effective_interaction_category_coverage"),
            unit="normalized_effective_categories",
            direction="higher",
            support={
                "observed": row.get("observed_category_count"),
                "taxonomy_size": len(TRANSITION_CATEGORIES),
            },
        ),
    }


def build_score_report(
    *,
    chunk_id: str,
    language: str,
    group_rows: list[dict],
    speaker_rows: list[dict],
    interaction_rows: list[dict],
    asr_rows: list[dict],
    audio_tag_rows: list[dict],
    failures: list[dict],
    manifest: dict,
) -> dict[str, object]:
    groups = {str(row.get("group")): row for row in group_rows}
    interactions = {str(row.get("group")): row for row in interaction_rows}
    asr = {str(row.get("group")): row for row in asr_rows}
    statuses = [row.get("status") for row in group_rows + interaction_rows]
    if statuses and all(value == "success" for value in statuses) and not failures:
        status = "partial"
    elif statuses and not any(value in {"success", "partial"} for value in statuses):
        status = "failed"
    else:
        status = "partial"
    shared = {
        group: _shared(groups.get(group), asr.get(group), language)
        for group in ("separation", "reconstruction", "expansion")
    }
    reconstruction = _reconstruction(interactions.get("reconstruction"))
    expansion = _expansion(
        interactions.get("expansion"), interactions.get("reconstruction")
    )
    audio_tag_summary = summarize_audio_tag_rows(audio_tag_rows)
    audio_tag_overall = audio_tag_summary["overall"]
    audio_tag_groups = audio_tag_summary["groups"]
    annotation_complete = (
        audio_tag_overall["evaluated_count"] > 0
        and audio_tag_overall["failed_count"] == 0
    )
    if (
        statuses
        and all(value == "success" for value in statuses)
        and not failures
        and annotation_complete
    ):
        status = "complete"
    return {
        "schema_version": 2,
        "chunk_id": chunk_id,
        "language": language,
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "stages": {
            "separation": {
                "shared_quality": shared["separation"],
                "interaction_descriptives": interactions.get("separation", {}),
            },
            "reconstruction": {
                "shared_quality": shared["reconstruction"],
                "fidelity": reconstruction,
            },
            "expansion": {
                "shared_quality": shared["expansion"],
                "realism_and_coverage": expansion,
            },
        },
        "comparisons": {
            "reconstruction_vs_separation": reconstruction,
            "expansion_vs_reconstruction": {
                name: value
                for name, value in expansion.items()
                if name
                in {
                    "turn_rate",
                    "backchannel_rate",
                    "interruption_rate",
                    "overlap_event_rate",
                    "transition_js_distance",
                }
            },
        },
        "annotation_quality": {
            "audio_tag_alignment_score": metric(
                audio_tag_overall["mean"],
                unit="score_1_to_5",
                direction="higher",
                support={
                    "evaluated_utterances": audio_tag_overall["evaluated_count"],
                    "failed_utterances": audio_tag_overall["failed_count"],
                    "score_counts": audio_tag_overall["score_counts"],
                },
                reason="No tagged utterance was successfully evaluated",
            ),
            "reconstruction_audio_tag_alignment_score": metric(
                audio_tag_groups["reconstruction"]["mean"],
                unit="score_1_to_5",
                direction="higher",
                support=audio_tag_groups["reconstruction"]["evaluated_count"],
                reason="No tagged reconstruction utterance was successfully evaluated",
            ),
            "expansion_audio_tag_alignment_score": metric(
                audio_tag_groups["expansion"]["mean"],
                unit="score_1_to_5",
                direction="higher",
                support=audio_tag_groups["expansion"]["evaluated_count"],
                reason="No tagged expansion utterance was successfully evaluated",
            ),
        },
        "coverage_notes": (
            []
            if annotation_complete
            else [
                "Audio-tag Alignment Score is unavailable or partial because no "
                "tagged utterance was evaluated successfully or some evaluations failed."
            ]
        ),
        "diagnostics": {
            "speaker_rows": speaker_rows,
            "interaction_rows": interaction_rows,
            "asr_rows": asr_rows,
            "audio_tag_summary": audio_tag_summary,
        },
        "failures": failures,
        "provenance": {
            "gpu_used": False,
            "local_asr_weights": False,
            "asr_provider": "openrouter",
            "manifest": manifest,
        },
    }


def metric_records(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def walk(value, path: list[str]):
        if isinstance(value, dict) and {
            "value",
            "unit",
            "direction",
            "status",
        }.issubset(value):
            rows.append(
                {
                    "chunk_id": report["chunk_id"],
                    "language": report["language"],
                    "metric": ".".join(path),
                    **value,
                }
            )
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, [*path, str(key)])

    for section in ("stages", "annotation_quality"):
        walk(report.get(section), [section])
    return rows


def render_artifacts(
    report: dict[str, object], event_rows: list[dict], audio_tag_rows: list[dict]
) -> dict[str, bytes]:
    records = metric_records(report)

    def jsonl(rows):
        payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
        return payload or b"\n"

    summary = io.StringIO()
    writer = csv.DictWriter(
        summary,
        fieldnames=(
            "chunk_id",
            "language",
            "metric",
            "value",
            "unit",
            "direction",
            "status",
        ),
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(records)
    asr_rows = report.get("diagnostics", {}).get("asr_rows", [])
    failures = report.get("failures", [])
    return {
        "score-report.json": (canonical_json(report) + "\n").encode("utf-8"),
        "metric-records.jsonl": jsonl(records),
        "score-summary.csv": summary.getvalue().encode("utf-8"),
        "event-matches.jsonl": jsonl(event_rows),
        "asr-transcripts.jsonl": jsonl(asr_rows),
        "audio-tag-scores.jsonl": jsonl(audio_tag_rows),
        "audio-tag-summary.json": (
            canonical_json(report["diagnostics"]["audio_tag_summary"]) + "\n"
        ).encode("utf-8"),
        "run-manifest.json": (
            canonical_json(report["provenance"]["manifest"]) + "\n"
        ).encode("utf-8"),
        "failures.jsonl": jsonl(failures),
    }
