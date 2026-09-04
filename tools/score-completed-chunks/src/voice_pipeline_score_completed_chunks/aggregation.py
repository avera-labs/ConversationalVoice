from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from .contracts import GROUPS

METRIC_FIELDS = (
    "nisqa_mos",
    "nisqa_noisiness",
    "nisqa_discontinuity",
    "nisqa_coloration",
    "nisqa_loudness",
    "dnsmos_ovrl",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_p808",
    "speaker_similarity",
    "same_speaker_margin",
)

PRIMARY_METRICS = ("nisqa_mos", "dnsmos_ovrl", "speaker_similarity")


def _valid_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def build_group_row(
    *,
    chunk_id: str,
    language: str,
    group: str,
    speaker_rows: list[dict],
) -> dict:
    valid = [row for row in speaker_rows if row.get("status") == "success"]
    row: dict[str, object] = {
        "chunk_id": chunk_id,
        "language": language,
        "group": group,
        "status": "success" if len(valid) == 2 else "partial" if valid else "failed",
        "valid_speaker_count": len(valid),
        "active_speech_duration_ms": sum(
            int(item.get("active_speech_duration_ms", 0)) for item in valid
        ),
        "speakers": [
            {
                "speaker_id": item.get("speaker_id"),
                "status": item.get("status"),
                "error_code": item.get("error_code"),
                **{
                    metric: item.get(metric)
                    for metric in PRIMARY_METRICS
                    if item.get(metric) is not None
                },
            }
            for item in speaker_rows
        ],
    }
    for metric in METRIC_FIELDS:
        measured = [item for item in valid if _valid_number(item.get(metric))]
        if not measured:
            row[metric] = None
            row[f"{metric}_min"] = None
            row[f"{metric}_speaker_abs_diff"] = None
            continue
        weights = np.asarray(
            [item["active_speech_duration_ms"] for item in measured], dtype=np.float64
        )
        values = np.asarray([item[metric] for item in measured], dtype=np.float64)
        row[metric] = float(np.average(values, weights=weights))
        row[f"{metric}_min"] = float(np.min(values))
        row[f"{metric}_speaker_abs_diff"] = (
            float(abs(values[0] - values[1])) if len(values) == 2 else None
        )
    return row


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _summarize_rows(rows: list[dict]) -> dict:
    status_counts = defaultdict(int)
    for row in rows:
        status_counts[str(row.get("status"))] += 1
    metrics: dict[str, object] = {}
    for metric in METRIC_FIELDS:
        valid = [row for row in rows if _valid_number(row.get(metric))]
        values = [float(row[metric]) for row in valid]
        weights = np.asarray(
            [row.get("active_speech_duration_ms", 0) for row in valid],
            dtype=np.float64,
        )
        micro = (
            float(np.average(np.asarray(values), weights=weights))
            if values and np.all(weights > 0)
            else None
        )
        metrics[metric] = {"macro": _describe(values), "micro_mean": micro}
    return {
        "chunk_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "metrics": metrics,
    }


def build_summary(rows: Iterable[dict]) -> dict:
    materialized = list(rows)
    by_group = {
        group: [row for row in materialized if row.get("group") == group]
        for group in GROUPS
    }
    languages = sorted({str(row.get("language")) for row in materialized})
    by_language = {
        language: {
            group: _summarize_rows(
                [row for row in by_group[group] if row.get("language") == language]
            )
            for group in by_group
        }
        for language in languages
    }
    keyed = {
        (str(row.get("chunk_id")), str(row.get("group"))): row for row in materialized
    }
    paired_deltas: dict[str, dict[str, object]] = {}
    for minuend, subtrahend in (
        ("reconstruction", "separation"),
        ("expansion", "reconstruction"),
        ("expansion", "separation"),
    ):
        paired: dict[str, object] = {}
        for metric in PRIMARY_METRICS:
            deltas: list[float] = []
            for chunk_id in {key[0] for key in keyed}:
                left = keyed.get((chunk_id, minuend))
                right = keyed.get((chunk_id, subtrahend))
                if (
                    left
                    and right
                    and left.get("status") == "success"
                    and right.get("status") == "success"
                    and _valid_number(left.get(metric))
                    and _valid_number(right.get(metric))
                ):
                    deltas.append(float(left[metric]) - float(right[metric]))
            paired[metric] = _describe(deltas)
        paired_deltas[f"paired_delta_{minuend}_minus_{subtrahend}"] = paired
    return {
        "schema_version": 1,
        "groups": {group: _summarize_rows(rows) for group, rows in by_group.items()},
        "languages": by_language,
        **paired_deltas,
    }
