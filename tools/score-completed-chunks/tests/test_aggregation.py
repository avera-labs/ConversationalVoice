from __future__ import annotations

import pytest

from voice_pipeline_score_completed_chunks.aggregation import (
    build_group_row,
    build_summary,
)


def speaker(speaker_id: int, duration: int, value: float) -> dict:
    return {
        "speaker_id": speaker_id,
        "status": "success",
        "error_code": None,
        "active_speech_duration_ms": duration,
        "nisqa_mos": value,
        "dnsmos_ovrl": value + 1,
        "speaker_similarity": value / 10,
    }


def test_group_metrics_are_active_duration_weighted() -> None:
    row = build_group_row(
        chunk_id="one",
        language="en",
        group="reconstruction",
        speaker_rows=[speaker(0, 1000, 2.0), speaker(1, 3000, 4.0)],
    )
    assert row["status"] == "success"
    assert row["nisqa_mos"] == pytest.approx(3.5)
    assert row["nisqa_mos_min"] == 2.0
    assert row["nisqa_mos_speaker_abs_diff"] == 2.0


def test_summary_reports_paired_delta() -> None:
    separation = build_group_row(
        chunk_id="one",
        language="en",
        group="separation",
        speaker_rows=[speaker(0, 1000, 1.0), speaker(1, 1000, 1.0)],
    )
    reconstruction = build_group_row(
        chunk_id="one",
        language="en",
        group="reconstruction",
        speaker_rows=[speaker(0, 1000, 2.0), speaker(1, 1000, 2.0)],
    )
    expansion = build_group_row(
        chunk_id="one",
        language="en",
        group="expansion",
        speaker_rows=[speaker(0, 1000, 3.0), speaker(1, 1000, 3.0)],
    )
    summary = build_summary([separation, reconstruction, expansion])
    delta = summary["paired_delta_expansion_minus_reconstruction"]["nisqa_mos"]
    assert delta["count"] == 1
    assert delta["mean"] == 1.0
    assert (
        summary["paired_delta_reconstruction_minus_separation"]["nisqa_mos"]["mean"]
        == 1.0
    )
    assert (
        summary["paired_delta_expansion_minus_separation"]["nisqa_mos"]["mean"] == 2.0
    )
