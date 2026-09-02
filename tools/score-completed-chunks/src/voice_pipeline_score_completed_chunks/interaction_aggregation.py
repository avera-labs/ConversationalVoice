from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .contracts import GROUPS
from .interaction_config import AUTOMATIC_ONLY, InteractionConfig
from .interaction_events import TRANSITION_CATEGORIES


def _number(value: object) -> float | None:
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _median(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return (
        float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
        if values
        else None
    )


def jensen_shannon_distance(left: list[float], right: list[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.size != right_array.size or left_array.size == 0:
        raise ValueError("distribution sizes disagree")
    if left_array.sum() <= 0 or right_array.sum() <= 0:
        return None
    left_array /= left_array.sum()
    right_array /= right_array.sum()
    middle = (left_array + right_array) / 2

    def divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / middle[mask])))

    return math.sqrt((divergence(left_array) + divergence(right_array)) / 2)


def wasserstein_distance(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    left_array = np.sort(np.asarray(left, dtype=np.float64))
    right_array = np.sort(np.asarray(right, dtype=np.float64))
    boundaries = np.sort(np.concatenate((left_array, right_array)))
    if boundaries.size < 2:
        return 0.0
    intervals = np.diff(boundaries)
    left_cdf = (
        np.searchsorted(left_array, boundaries[:-1], side="right") / left_array.size
    )
    right_cdf = (
        np.searchsorted(right_array, boundaries[:-1], side="right") / right_array.size
    )
    return float(np.sum(np.abs(left_cdf - right_cdf) * intervals))


def _aggregate_stage(rows: list[dict], config: InteractionConfig) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "success"]
    counts = Counter()
    eligible = 0
    overlap_count = 0
    turn_count = 0
    backchannel_count = 0
    interruption_count = 0
    overlap_event_count = 0
    duration_ms = 0
    active_ms = 0
    nonverbal_numerator = 0
    nonverbal_denominator = 0
    nonverbal_events = 0
    overlap_durations: list[float] = []
    gaps: list[float] = []
    for row in successful:
        eligible += int(row.get("eligible_transition_count", 0))
        raw_counts = row.get("transition_counts")
        if isinstance(raw_counts, dict):
            for category in TRANSITION_CATEGORIES:
                counts[category] += int(raw_counts.get(category, 0))
        overlap_count += int(row.get("overlap_transition_count", 0))
        turn_count += int(row.get("turn_count", 0))
        overlap_event_count += int(row.get("overlap_event_count", 0))
        if isinstance(raw_counts, dict):
            backchannel_count += int(raw_counts.get("backchannel", 0))
            interruption_count += int(raw_counts.get("interruption", 0))
        duration_ms += int(row.get("conversation_duration_ms", 0))
        active_ms += int(row.get("active_speech_duration_ms", 0))
        if isinstance(row.get("observed_nonverbal_utterance_count"), int):
            nonverbal_numerator += int(row["observed_nonverbal_utterance_count"])
            nonverbal_denominator += int(row.get("eligible_utterance_count", 0))
            nonverbal_events += int(row.get("observed_nonverbal_event_count", 0))
        overlap_durations.extend(
            float(value) for value in row.get("overlap_durations_ms", [])
        )
        gaps.extend(float(value) for value in row.get("inter_turn_gaps_ms", []))
    probabilities = (
        [
            counts[category] / eligible
            for category in TRANSITION_CATEGORIES
            if counts[category]
        ]
        if eligible
        else []
    )
    entropy = (
        math.exp(-sum(value * math.log(value) for value in probabilities))
        if probabilities
        else 0.0
    )
    support = sorted(
        category
        for category in TRANSITION_CATEGORIES
        if counts[category] >= config.category_minimum_support
    )
    return {
        "discovered_chunk_count": len(rows),
        "successful_chunk_count": len(successful),
        "eligible_transition_count": eligible,
        "transition_counts": {
            category: counts[category] for category in TRANSITION_CATEGORIES
        },
        "transition_rates": {
            category: _ratio(counts[category], eligible)
            for category in TRANSITION_CATEGORIES
        },
        "overlap_transition_count": overlap_count,
        "overlap_transition_rate": _ratio(overlap_count, eligible),
        "turn_rate_per_minute": _ratio(turn_count, duration_ms / 60_000),
        "backchannel_rate_per_minute": _ratio(backchannel_count, duration_ms / 60_000),
        "interruption_rate_per_minute": _ratio(
            interruption_count, duration_ms / 60_000
        ),
        "overlap_event_rate_per_minute": _ratio(
            overlap_event_count, duration_ms / 60_000
        ),
        "overlap_density_per_conversation_minute": _ratio(
            overlap_count, duration_ms / 60_000
        ),
        "nonverbal_utterance_rate": _ratio(nonverbal_numerator, nonverbal_denominator),
        "nonverbal_density_per_active_speech_minute": _ratio(
            nonverbal_events, active_ms / 60_000
        ),
        "interaction_entropy_effective_categories": entropy,
        "effective_interaction_category_coverage": (
            entropy / len(TRANSITION_CATEGORIES) if eligible else None
        ),
        "category_support": support,
        "overlap_durations_ms": overlap_durations,
        "inter_turn_gaps_ms": gaps,
    }


def _aggregate_reconstruction(rows: list[dict]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "success"]
    turn_numerator = sum(int(row.get("turn_preserved_count", 0)) for row in successful)
    turn_denominator = sum(int(row.get("turn_source_count", 0)) for row in successful)
    overlap_numerator = sum(
        int(row.get("overlap_preserved_count", 0)) for row in successful
    )
    overlap_denominator = sum(
        int(row.get("overlap_source_count", 0)) for row in successful
    )
    backchannel_numerator = sum(
        int(row.get("backchannel_preserved_count", 0)) for row in successful
    )
    backchannel_denominator = sum(
        int(row.get("backchannel_source_count", 0)) for row in successful
    )
    gap_errors = [
        float(value) for row in successful for value in row.get("gap_errors_ms", [])
    ]
    overlap_errors = [
        float(value)
        for row in successful
        for value in row.get("overlap_relative_errors", [])
    ]
    event_metrics: dict[str, object] = {}
    for prefix in ("turn_event", "overlap_event", "backchannel_event"):
        true_positive = sum(int(row.get(f"{prefix}_tp", 0)) for row in successful)
        false_positive = sum(int(row.get(f"{prefix}_fp", 0)) for row in successful)
        false_negative = sum(int(row.get(f"{prefix}_fn", 0)) for row in successful)
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        event_metrics[prefix] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision is not None
                and recall is not None
                and precision + recall > 0
                else None
            ),
        }
    return {
        **event_metrics,
        "turn_preserved_count": turn_numerator,
        "turn_source_count": turn_denominator,
        "turn_preservation": _ratio(turn_numerator, turn_denominator),
        "overlap_preserved_count": overlap_numerator,
        "overlap_source_count": overlap_denominator,
        "overlap_preservation": _ratio(overlap_numerator, overlap_denominator),
        "overlap_error_median": _median(overlap_errors),
        "overlap_error_mean": _mean(overlap_errors),
        "overlap_error_p90": _percentile(overlap_errors, 90),
        "backchannel_preserved_count": backchannel_numerator,
        "backchannel_source_count": backchannel_denominator,
        "backchannel_preservation": _ratio(
            backchannel_numerator, backchannel_denominator
        ),
        "gap_error_median_ms": _median(gap_errors),
        "gap_error_mae_ms": _mean(gap_errors),
        "gap_error_p90_ms": _percentile(gap_errors, 90),
        "gap_errors_ms": gap_errors,
        "overlap_relative_errors": overlap_errors,
    }


def _aggregate_scope(rows: list[dict], config: InteractionConfig) -> dict[str, object]:
    groups = {
        group: _aggregate_stage(
            [row for row in rows if row.get("group") == group], config
        )
        for group in GROUPS
    }
    comparisons: dict[str, object] = {}
    comparison_pairs = {
        "reconstruction_vs_separation": (
            groups["reconstruction"],
            groups["separation"],
        ),
        "expansion_vs_reconstruction": (
            groups["expansion"],
            groups["reconstruction"],
        ),
        # Keep the noisier source comparison as a secondary diagnostic.
        "expansion_vs_separation": (groups["expansion"], groups["separation"]),
    }
    for comparison_name, (current, reference) in comparison_pairs.items():
        reference_support = set(reference["category_support"])
        current_support = set(current["category_support"])
        comparisons[comparison_name] = {
            "transition_js_distance": jensen_shannon_distance(
                [
                    reference["transition_counts"][category]
                    for category in TRANSITION_CATEGORIES
                ],
                [
                    current["transition_counts"][category]
                    for category in TRANSITION_CATEGORIES
                ],
            ),
            "category_support_retention": _ratio(
                len(reference_support.intersection(current_support)),
                len(reference_support),
            ),
            "overlap_duration_wasserstein_ms": wasserstein_distance(
                reference["overlap_durations_ms"], current["overlap_durations_ms"]
            ),
            "inter_turn_gap_wasserstein_ms": wasserstein_distance(
                reference["inter_turn_gaps_ms"], current["inter_turn_gaps_ms"]
            ),
            "transition_rate_deltas": {
                category: (
                    current["transition_rates"][category]
                    - reference["transition_rates"][category]
                    if current["transition_rates"][category] is not None
                    and reference["transition_rates"][category] is not None
                    else None
                )
                for category in TRANSITION_CATEGORIES
            },
            "nonverbal_utterance_rate_delta": (
                current["nonverbal_utterance_rate"]
                - reference["nonverbal_utterance_rate"]
                if current["nonverbal_utterance_rate"] is not None
                and reference["nonverbal_utterance_rate"] is not None
                else None
            ),
        }
    return {
        "groups": groups,
        "comparisons": comparisons,
        "reconstruction_preservation": _aggregate_reconstruction(
            [row for row in rows if row.get("group") == "reconstruction"]
        ),
    }


def _binary_metrics(
    rows: list[dict], declared_key: str, observed_key: str
) -> dict[str, object]:
    eligible = [row for row in rows if isinstance(row.get(observed_key), bool)]
    true_positive = sum(
        bool(row.get(declared_key)) and bool(row[observed_key]) for row in eligible
    )
    false_positive = sum(
        bool(row.get(declared_key)) and not bool(row[observed_key]) for row in eligible
    )
    false_negative = sum(
        not bool(row.get(declared_key)) and bool(row[observed_key]) for row in eligible
    )
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "eligible_count": len(eligible),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_declared_observed(rows: Iterable[dict]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "success"]
    boundaries = [
        float(value)
        for row in successful
        for key in (
            "start_boundary_absolute_error_ms",
            "end_boundary_absolute_error_ms",
        )
        if (value := _number(row.get(key))) is not None
    ]
    tag_scores = [
        value
        for row in successful
        if (value := _number(row.get("audio_tag_score"))) is not None
    ]
    return {
        "overlap": _binary_metrics(successful, "declared_overlap", "observed_overlap"),
        "backchannel": _binary_metrics(
            successful, "declared_backchannel", "observed_backchannel"
        ),
        "paralinguistic": _binary_metrics(
            successful, "declared_paralinguistic", "observed_nonverbal"
        ),
        "boundary_mae_ms": _mean(boundaries),
        "audio_tag_score_mean": _mean(tag_scores),
        "audio_tag_score_count": len(tag_scores),
    }


def _flatten_claim_metrics(scope: dict[str, object]) -> dict[str, float]:
    output: dict[str, float] = {}
    for group, values in scope["groups"].items():
        for category, rate in values["transition_rates"].items():
            if (number := _number(rate)) is not None:
                output[f"group.{group}.transition_rate.{category}"] = number
        for name in (
            "overlap_transition_rate",
            "turn_rate_per_minute",
            "backchannel_rate_per_minute",
            "interruption_rate_per_minute",
            "overlap_event_rate_per_minute",
            "nonverbal_utterance_rate",
            "effective_interaction_category_coverage",
        ):
            if (number := _number(values.get(name))) is not None:
                output[f"group.{group}.{name}"] = number
    for comparison, values in scope["comparisons"].items():
        for name in (
            "transition_js_distance",
            "category_support_retention",
            "overlap_duration_wasserstein_ms",
            "inter_turn_gap_wasserstein_ms",
        ):
            if (number := _number(values.get(name))) is not None:
                output[f"comparison.{comparison}.{name}"] = number
        for category, delta in values["transition_rate_deltas"].items():
            if (number := _number(delta)) is not None:
                output[f"comparison.{comparison}.transition_rate_delta.{category}"] = (
                    number
                )
        if (
            number := _number(values.get("nonverbal_utterance_rate_delta"))
        ) is not None:
            output[f"comparison.{comparison}.nonverbal_utterance_rate_delta"] = number
    preservation = scope["reconstruction_preservation"]
    for event_name in ("turn_event", "overlap_event", "backchannel_event"):
        event = preservation.get(event_name, {})
        if not isinstance(event, dict):
            continue
        for name in ("precision", "recall", "f1"):
            if (number := _number(event.get(name))) is not None:
                output[f"preservation.{event_name}_{name}"] = number
    for name in (
        "turn_preservation",
        "overlap_preservation",
        "overlap_error_median",
        "overlap_error_mean",
        "overlap_error_p90",
        "backchannel_preservation",
        "gap_error_median_ms",
        "gap_error_mae_ms",
        "gap_error_p90_ms",
    ):
        if (number := _number(preservation.get(name))) is not None:
            output[f"preservation.{name}"] = number
    return output


def _bootstrap_scope(
    rows: list[dict],
    config: InteractionConfig,
    rng: np.random.Generator,
    *,
    stratify_language: bool = False,
) -> dict[str, dict[str, float | int | None]]:
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cluster = str(row.get("source_cluster_id") or row.get("chunk_id"))
        by_cluster[cluster].append(row)
    clusters = sorted(by_cluster)
    if not clusters:
        return {}
    observed = _flatten_claim_metrics(_aggregate_scope(rows, config))
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(config.bootstrap_samples):
        if stratify_language:
            cluster_languages = {
                cluster: str(by_cluster[cluster][0].get("language"))
                for cluster in clusters
            }
            selected_values: list[str] = []
            for language in sorted(set(cluster_languages.values())):
                stratum = [
                    cluster
                    for cluster in clusters
                    if cluster_languages[cluster] == language
                ]
                selected_values.extend(
                    str(value)
                    for value in rng.choice(stratum, size=len(stratum), replace=True)
                )
            selected = selected_values
        else:
            selected = [
                str(value)
                for value in rng.choice(clusters, size=len(clusters), replace=True)
            ]
        replicate = [row for cluster in selected for row in by_cluster[str(cluster)]]
        for metric, value in _flatten_claim_metrics(
            _aggregate_scope(replicate, config)
        ).items():
            samples[metric].append(value)
    result: dict[str, dict[str, float | int | None]] = {}
    for metric, estimate in observed.items():
        values = samples.get(metric, [])
        interval: dict[str, float | int | None] = {
            "estimate": estimate,
            "ci_lower": float(np.percentile(values, 2.5)) if values else None,
            "ci_upper": float(np.percentile(values, 97.5)) if values else None,
            "replicate_count": len(values),
            "cluster_count": len(clusters),
        }
        result[metric] = interval
    return result


def build_interaction_summary(
    score_rows: Iterable[dict],
    declared_rows: Iterable[dict],
    *,
    config: InteractionConfig,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    scores = list(score_rows)
    declared = list(declared_rows)
    successful = [row for row in scores if row.get("status") == "success"]
    languages = sorted({str(row.get("language")) for row in successful})
    global_scope = _aggregate_scope(successful, config)
    language_scopes = {
        language: _aggregate_scope(
            [row for row in successful if row.get("language") == language], config
        )
        for language in languages
    }
    rng = np.random.default_rng(config.seed)
    bootstrap: dict[str, object] = {
        "schema_version": 1,
        "seed": config.seed,
        "bootstrap_samples": config.bootstrap_samples,
        "cluster_unit": "source_cluster_id",
        "global": _bootstrap_scope(successful, config, rng, stratify_language=True),
        "languages": {
            language: _bootstrap_scope(
                [row for row in successful if row.get("language") == language],
                config,
                rng,
            )
            for language in languages
        },
    }
    paired_rows: list[dict[str, object]] = []
    for scope_name, scope in [("global", global_scope), *language_scopes.items()]:
        for comparison, values in scope["comparisons"].items():
            for metric, value in values.items():
                if metric == "transition_rate_deltas":
                    for category, delta in value.items():
                        paired_rows.append(
                            {
                                "scope": scope_name,
                                "comparison": comparison,
                                "metric": f"transition_rate_delta.{category}",
                                "estimate": delta,
                            }
                        )
                else:
                    paired_rows.append(
                        {
                            "scope": scope_name,
                            "comparison": comparison,
                            "metric": metric,
                            "estimate": value,
                        }
                    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "detector_fingerprint": (
            successful[0].get("detector_fingerprint") if successful else None
        ),
        "provisional_automatic_only": AUTOMATIC_ONLY,
        "estimand": "event-weighted-micro",
        "cluster_unit": "source_cluster_id",
        "category_minimum_support": config.category_minimum_support,
        "global": global_scope,
        "languages": language_scopes,
        "declared_vs_observed": summarize_declared_observed(declared),
        "counts": {
            "score_rows": len(scores),
            "successful_score_rows": len(successful),
            "failed_score_rows": len(scores) - len(successful),
            "paired_chunk_count": len(
                {
                    row.get("chunk_id")
                    for row in successful
                    if all(
                        any(
                            candidate.get("chunk_id") == row.get("chunk_id")
                            and candidate.get("group") == group
                            for candidate in successful
                        )
                        for group in GROUPS
                    )
                }
            ),
        },
    }
    return summary, bootstrap, paired_rows
