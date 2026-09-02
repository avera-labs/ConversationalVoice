import json

from voice_pipeline_score_completed_chunks.reporting import (
    build_score_report,
    metric_records,
    render_artifacts,
)


def test_report_exposes_stable_nested_metric_objects() -> None:
    report = build_score_report(
        chunk_id="chunk-1",
        language="en",
        group_rows=[
            {
                "group": group,
                "status": "success",
                "nisqa_mos": 4.0,
                "dnsmos_ovrl": 3.5,
                "speaker_similarity": 0.8,
                "same_speaker_margin": 0.2,
                "valid_speaker_count": 2,
            }
            for group in ("separation", "reconstruction", "expansion")
        ],
        speaker_rows=[],
        interaction_rows=[
            {
                "group": "separation",
                "status": "success",
                "transition_counts": {
                    "backchannel": 1,
                    "interruption": 0,
                    "other_overlap": 0,
                    "clean_transition": 2,
                    "delayed_other": 0,
                },
            },
            {
                "group": "reconstruction",
                "status": "success",
                "duration_ratio": 1.0,
                "duration_log_error": 0.0,
                "turn_event_f1": 1.0,
                "overlap_event_f1": 1.0,
                "backchannel_event_f1": 1.0,
                "transition_counts": {
                    "backchannel": 1,
                    "interruption": 0,
                    "other_overlap": 0,
                    "clean_transition": 2,
                    "delayed_other": 0,
                },
            },
            {
                "group": "expansion",
                "status": "success",
                "expansion_factor": 2.0,
                "turn_rate_per_minute": 10.0,
                "backchannel_rate_per_minute": 2.0,
                "interruption_rate_per_minute": 1.0,
                "overlap_event_rate_per_minute": 3.0,
                "transition_counts": {
                    "backchannel": 1,
                    "interruption": 0,
                    "other_overlap": 0,
                    "clean_transition": 2,
                    "delayed_other": 0,
                },
                "eligible_transition_count": 3,
                "effective_interaction_category_coverage": 0.4,
            },
        ],
        asr_rows=[
            {
                "group": group,
                "status": "success",
                "error": {"value": 0.1, "reference_unit_count": 10},
            }
            for group in ("separation", "reconstruction", "expansion")
        ],
        audio_tag_rows=[
            {
                "group": "reconstruction",
                "language": "en",
                "status": "success",
                "score": 4,
            },
            {
                "group": "expansion",
                "language": "en",
                "status": "success",
                "score": 5,
            },
        ],
        failures=[],
        manifest={"schema_version": 2},
    )
    assert report["status"] == "complete"
    assert (
        report["stages"]["expansion"]["shared_quality"]["wer"]["direction"] == "lower"
    )
    assert "target" not in report["stages"]["reconstruction"]["fidelity"][
        "duration_ratio"
    ]
    assert (
        report["annotation_quality"]["audio_tag_alignment_score"]["value"] == 4.5
    )
    assert "target" not in report["annotation_quality"]["audio_tag_alignment_score"]
    expansion_comparison = report["comparisons"]["expansion_vs_reconstruction"]
    assert expansion_comparison["transition_js_distance"]["value"] == 0.0
    assert expansion_comparison["transition_js_distance"]["reference"].startswith(
        "paired reconstruction"
    )
    records = metric_records(report)
    assert any(row["metric"].endswith("transition_js_distance") for row in records)
    artifacts = render_artifacts(report, [], [])
    assert set(artifacts) == {
        "score-report.json",
        "metric-records.jsonl",
        "score-summary.csv",
        "event-matches.jsonl",
        "asr-transcripts.jsonl",
        "audio-tag-scores.jsonl",
        "audio-tag-summary.json",
        "run-manifest.json",
        "failures.jsonl",
    }
    assert json.loads(artifacts["score-report.json"])["schema_version"] == 2
