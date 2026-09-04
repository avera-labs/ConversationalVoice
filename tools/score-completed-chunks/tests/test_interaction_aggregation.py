from __future__ import annotations

import pytest

from voice_pipeline_score_completed_chunks.interaction_aggregation import (
    build_interaction_summary,
    jensen_shannon_distance,
    wasserstein_distance,
)
from voice_pipeline_score_completed_chunks.interaction_config import InteractionConfig
from voice_pipeline_score_completed_chunks.interaction_events import (
    TRANSITION_CATEGORIES,
)


def score(group: str, *, clean: int, overlap: int) -> dict:
    counts = {category: 0 for category in TRANSITION_CATEGORIES}
    counts["clean_transition"] = clean
    counts["other_overlap"] = overlap
    eligible = clean + overlap
    row = {
        "chunk_id": "one",
        "source_cluster_id": "source-one",
        "language": "en",
        "group": group,
        "status": "success",
        "detector_fingerprint": "fingerprint",
        "eligible_transition_count": eligible,
        "transition_counts": counts,
        "overlap_transition_count": overlap,
        "conversation_duration_ms": 60_000,
        "active_speech_duration_ms": 30_000,
        "overlap_durations_ms": [200.0] * overlap,
        "inter_turn_gaps_ms": [300.0] * clean,
        "eligible_utterance_count": 10,
        "observed_nonverbal_utterance_count": 1,
        "observed_nonverbal_event_count": 1,
    }
    if group == "reconstruction":
        row.update(
            {
                "turn_preserved_count": 9,
                "turn_source_count": 10,
                "overlap_preserved_count": overlap,
                "overlap_source_count": overlap,
                "overlap_relative_errors": [0.1] * overlap,
                "backchannel_preserved_count": 0,
                "backchannel_source_count": 0,
                "gap_errors_ms": [50.0] * clean,
            }
        )
    return row


def test_distribution_distances() -> None:
    assert jensen_shannon_distance([1, 0], [1, 0]) == 0.0
    assert jensen_shannon_distance([1, 0], [0, 1]) == 1.0
    assert wasserstein_distance([0, 1], [1, 2]) == 1.0


def test_interaction_summary_bootstraps_source_clusters() -> None:
    config = InteractionConfig(
        category_minimum_support=1,
        bootstrap_samples=20,
        seed=7,
    )
    rows = [
        score("separation", clean=8, overlap=2),
        score("reconstruction", clean=8, overlap=2),
        score("expansion", clean=7, overlap=3),
    ]
    summary, bootstrap, paired = build_interaction_summary(rows, [], config=config)
    preservation = summary["global"]["reconstruction_preservation"]
    assert preservation["turn_preservation"] == 0.9
    assert not any("iou" in key.casefold() for key in preservation)
    assert (
        summary["global"]["comparisons"]["reconstruction_vs_separation"][
            "transition_js_distance"
        ]
        == 0.0
    )
    assert summary["global"]["comparisons"]["expansion_vs_reconstruction"][
        "transition_js_distance"
    ] == pytest.approx(
        summary["global"]["comparisons"]["expansion_vs_separation"][
            "transition_js_distance"
        ]
    )
    turn = bootstrap["global"]["preservation.turn_preservation"]
    assert turn["cluster_count"] == 1
    assert turn["replicate_count"] == 20
    assert not any("iou" in key.casefold() for key in bootstrap["global"])
    assert paired
    assert summary["global"]["groups"]["expansion"][
        "overlap_transition_rate"
    ] == pytest.approx(0.3)
