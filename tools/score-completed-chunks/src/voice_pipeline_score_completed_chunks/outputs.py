from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .aggregation import METRIC_FIELDS
from .audio_tag_accuracy import EVALUATED_GROUPS


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


class JsonlAppender:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def write(self, value: dict) -> None:
        self.handle.write(canonical_json(value) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


class RunOutputs:
    def __init__(self, directory: Path, *, resume: bool):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.speaker_path = directory / "speaker-scores.jsonl"
        self.group_path = directory / "chunk-group-scores.jsonl"
        self.audio_tag_path = directory / "audio-tag-scores.jsonl"
        self.interaction_event_path = directory / "interaction-events.jsonl"
        self.interaction_score_path = directory / "chunk-interaction-scores.jsonl"
        self.interaction_declared_path = (
            directory / "interaction-declared-observed.jsonl"
        )
        self.failure_path = directory / "failures.jsonl"
        if not resume and any(
            path.exists()
            for path in (
                self.speaker_path,
                self.group_path,
                self.audio_tag_path,
                self.interaction_event_path,
                self.interaction_score_path,
                self.interaction_declared_path,
                self.failure_path,
            )
        ):
            raise RuntimeError(
                "output directory already contains a scoring run; use --resume"
            )
        existing = read_jsonl(self.speaker_path) if resume else []
        latest = {
            (row.get("chunk_id"), row.get("group"), row.get("speaker_id")): row
            for row in existing
        }
        if resume:
            self.speaker_path.write_text(
                "".join(canonical_json(row) + "\n" for row in latest.values()),
                encoding="utf-8",
            )
        existing_audio_tags = read_jsonl(self.audio_tag_path) if resume else []
        latest_audio_tags = {
            (
                row.get("chunk_id"),
                row.get("group"),
                row.get("transcript_index"),
            ): row
            for row in existing_audio_tags
        }
        if resume:
            self.audio_tag_path.write_text(
                "".join(
                    canonical_json(row) + "\n" for row in latest_audio_tags.values()
                ),
                encoding="utf-8",
            )
        self.existing_speakers = latest
        self.existing_audio_tags = latest_audio_tags
        existing_interaction_events = (
            read_jsonl(self.interaction_event_path) if resume else []
        )
        self.existing_interaction_events = {
            (row.get("chunk_id"), row.get("group"), row.get("event_id")): row
            for row in existing_interaction_events
        }
        existing_interaction_scores = (
            read_jsonl(self.interaction_score_path) if resume else []
        )
        self.existing_interaction_scores = {
            (row.get("chunk_id"), row.get("group")): row
            for row in existing_interaction_scores
        }
        existing_interaction_declared = (
            read_jsonl(self.interaction_declared_path) if resume else []
        )
        self.existing_interaction_declared = {
            (row.get("chunk_id"), row.get("group"), row.get("utterance_index")): row
            for row in existing_interaction_declared
        }
        if resume:
            for path, values in (
                (
                    self.interaction_event_path,
                    self.existing_interaction_events.values(),
                ),
                (
                    self.interaction_score_path,
                    self.existing_interaction_scores.values(),
                ),
                (
                    self.interaction_declared_path,
                    self.existing_interaction_declared.values(),
                ),
            ):
                path.write_text(
                    "".join(canonical_json(row) + "\n" for row in values),
                    encoding="utf-8",
                )
        self.group_path.write_text("", encoding="utf-8")
        self.speaker = JsonlAppender(self.speaker_path)
        self.group = JsonlAppender(self.group_path)
        self.audio_tag = JsonlAppender(self.audio_tag_path)
        self.interaction_event = JsonlAppender(self.interaction_event_path)
        self.interaction_score = JsonlAppender(self.interaction_score_path)
        self.interaction_declared = JsonlAppender(self.interaction_declared_path)
        self.failure = JsonlAppender(self.failure_path)
        self.group_rows: list[dict] = []
        self.audio_tag_rows: list[dict] = []
        self.interaction_event_rows: list[dict] = []
        self.interaction_score_rows: list[dict] = []
        self.interaction_declared_rows: list[dict] = []

    def write_speaker(self, row: dict) -> None:
        self.speaker.write(row)
        self.existing_speakers[(row["chunk_id"], row["group"], row["speaker_id"])] = row

    def write_group(self, row: dict) -> None:
        self.group.write(row)
        self.group_rows.append(row)

    def write_audio_tag(self, row: dict) -> None:
        self.audio_tag.write(row)
        key = (row["chunk_id"], row["group"], row["transcript_index"])
        self.existing_audio_tags[key] = row
        self.audio_tag_rows.append(row)

    def write_failure(self, row: dict) -> None:
        self.failure.write(row)

    def write_interaction_event(self, row: dict) -> None:
        key = (row["chunk_id"], row["group"], row["event_id"])
        previous = self.existing_interaction_events.get(key)
        if previous and previous.get("resume_key") == row.get("resume_key"):
            self.interaction_event_rows.append(previous)
            return
        self.interaction_event.write(row)
        self.existing_interaction_events[key] = row
        self.interaction_event_rows.append(row)

    def write_interaction_score(self, row: dict) -> None:
        key = (row["chunk_id"], row["group"])
        previous = self.existing_interaction_scores.get(key)
        if (
            previous
            and previous.get("status") == "success"
            and previous.get("resume_key") == row.get("resume_key")
        ):
            self.interaction_score_rows.append(previous)
            return
        for event_key in [
            candidate
            for candidate in self.existing_interaction_events
            if candidate[:2] == key
        ]:
            self.existing_interaction_events.pop(event_key)
        for declared_key in [
            candidate
            for candidate in self.existing_interaction_declared
            if candidate[:2] == key
        ]:
            self.existing_interaction_declared.pop(declared_key)
        self.interaction_score.write(row)
        self.existing_interaction_scores[key] = row
        self.interaction_score_rows.append(row)

    def write_interaction_declared(self, row: dict) -> None:
        key = (row["chunk_id"], row["group"], row["utterance_index"])
        previous = self.existing_interaction_declared.get(key)
        if previous and previous.get("resume_key") == row.get("resume_key"):
            self.interaction_declared_rows.append(previous)
            return
        self.interaction_declared.write(row)
        self.existing_interaction_declared[key] = row
        self.interaction_declared_rows.append(row)

    def write_interaction_summary(
        self,
        summary: dict,
        bootstrap: dict,
        paired_rows: list[dict],
    ) -> None:
        atomic_json(self.directory / "interaction-summary.json", summary)
        atomic_json(self.directory / "interaction-bootstrap.json", bootstrap)
        with (self.directory / "interaction-paired-deltas.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = ["scope", "comparison", "metric", "estimate"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(paired_rows)
        with (self.directory / "interaction-summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = [
                "scope",
                "group",
                "metric",
                "numerator",
                "denominator",
                "estimate",
                "ci_lower",
                "ci_upper",
                "provisional_automatic_only",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            bootstrap_scopes = {
                "global": bootstrap.get("global", {}),
                **bootstrap.get("languages", {}),
            }
            scopes = {"global": summary["global"], **summary["languages"]}
            for scope_name, scope in scopes.items():
                confidence = bootstrap_scopes.get(scope_name, {})
                for group, values in scope["groups"].items():
                    for category, estimate in values["transition_rates"].items():
                        metric = f"transition_rate.{category}"
                        ci = confidence.get(
                            f"group.{group}.transition_rate.{category}", {}
                        )
                        writer.writerow(
                            {
                                "scope": scope_name,
                                "group": group,
                                "metric": metric,
                                "numerator": values["transition_counts"][category],
                                "denominator": values["eligible_transition_count"],
                                "estimate": estimate,
                                "ci_lower": ci.get("ci_lower"),
                                "ci_upper": ci.get("ci_upper"),
                                "provisional_automatic_only": summary[
                                    "provisional_automatic_only"
                                ],
                            }
                        )
                preservation = scope["reconstruction_preservation"]
                for metric, numerator_key, denominator_key in (
                    ("turn_preservation", "turn_preserved_count", "turn_source_count"),
                    (
                        "overlap_preservation",
                        "overlap_preserved_count",
                        "overlap_source_count",
                    ),
                    (
                        "backchannel_preservation",
                        "backchannel_preserved_count",
                        "backchannel_source_count",
                    ),
                ):
                    ci = confidence.get(f"preservation.{metric}", {})
                    writer.writerow(
                        {
                            "scope": scope_name,
                            "group": "reconstruction",
                            "metric": metric,
                            "numerator": preservation[numerator_key],
                            "denominator": preservation[denominator_key],
                            "estimate": preservation[metric],
                            "ci_lower": ci.get("ci_lower"),
                            "ci_upper": ci.get("ci_upper"),
                            "provisional_automatic_only": summary[
                                "provisional_automatic_only"
                            ],
                        }
                    )
                for metric in (
                    "gap_error_median_ms",
                    "gap_error_mae_ms",
                    "gap_error_p90_ms",
                    "overlap_error_median",
                    "overlap_error_mean",
                    "overlap_error_p90",
                ):
                    ci = confidence.get(f"preservation.{metric}", {})
                    writer.writerow(
                        {
                            "scope": scope_name,
                            "group": "reconstruction",
                            "metric": metric,
                            "numerator": None,
                            "denominator": None,
                            "estimate": preservation.get(metric),
                            "ci_lower": ci.get("ci_lower"),
                            "ci_upper": ci.get("ci_upper"),
                            "provisional_automatic_only": summary[
                                "provisional_automatic_only"
                            ],
                        }
                    )
                for comparison, values in scope["comparisons"].items():
                    compared_group = comparison.removesuffix("_vs_separation")
                    for metric in (
                        "transition_js_distance",
                        "category_support_retention",
                        "overlap_duration_wasserstein_ms",
                        "inter_turn_gap_wasserstein_ms",
                        "nonverbal_utterance_rate_delta",
                    ):
                        ci = confidence.get(f"comparison.{comparison}.{metric}", {})
                        writer.writerow(
                            {
                                "scope": scope_name,
                                "group": compared_group,
                                "metric": metric,
                                "numerator": None,
                                "denominator": None,
                                "estimate": values.get(metric),
                                "ci_lower": ci.get("ci_lower"),
                                "ci_upper": ci.get("ci_upper"),
                                "provisional_automatic_only": summary[
                                    "provisional_automatic_only"
                                ],
                            }
                        )

    def write_summary(self, summary: dict) -> None:
        atomic_json(self.directory / "summary.json", summary)
        with (self.directory / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "scope",
                    "language",
                    "group",
                    "metric",
                    "count",
                    "mean",
                    "std",
                    "median",
                    "p05",
                    "p25",
                    "p75",
                    "p95",
                    "min",
                    "max",
                    "micro_mean",
                ],
            )
            writer.writeheader()
            for language, groups in [
                ("", summary["groups"]),
                *summary["languages"].items(),
            ]:
                scope = "global" if language == "" else "language"
                for group, group_summary in groups.items():
                    for metric in METRIC_FIELDS:
                        values = group_summary["metrics"][metric]
                        writer.writerow(
                            {
                                "scope": scope,
                                "language": language,
                                "group": group,
                                "metric": metric,
                                **values["macro"],
                                "micro_mean": values["micro_mean"],
                            }
                        )

    def write_audio_tag_summary(self, summary: dict) -> None:
        atomic_json(self.directory / "audio-tag-summary.json", summary)
        with (self.directory / "audio-tag-summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = [
                "scope",
                "language",
                "group",
                "utterance_count",
                "evaluated_count",
                "not_applicable_count",
                "failed_count",
                "score_1_count",
                "score_2_count",
                "score_3_count",
                "score_4_count",
                "score_5_count",
                "score_1_proportion",
                "score_2_proportion",
                "score_3_proportion",
                "score_4_proportion",
                "score_5_proportion",
                "mean",
                "median",
                "well_expressed_rate_at_least_4",
                "perfect_match_rate",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            def write_row(
                *, scope: str, language: str, group: str, values: dict
            ) -> None:
                writer.writerow(
                    {
                        "scope": scope,
                        "language": language,
                        "group": group,
                        **{
                            field: values[field]
                            for field in (
                                "utterance_count",
                                "evaluated_count",
                                "not_applicable_count",
                                "failed_count",
                                "mean",
                                "median",
                                "well_expressed_rate_at_least_4",
                                "perfect_match_rate",
                            )
                        },
                        **{
                            f"score_{score}_count": values["score_counts"][str(score)]
                            for score in range(1, 6)
                        },
                        **{
                            f"score_{score}_proportion": values["score_proportions"][
                                str(score)
                            ]
                            for score in range(1, 6)
                        },
                    }
                )

            write_row(
                scope="global",
                language="",
                group="",
                values=summary["overall"],
            )
            for group in EVALUATED_GROUPS:
                write_row(
                    scope="group",
                    language="",
                    group=group,
                    values=summary["groups"][group],
                )
            for language, language_summary in summary["languages"].items():
                write_row(
                    scope="language",
                    language=language,
                    group="",
                    values=language_summary["overall"],
                )
                for group in EVALUATED_GROUPS:
                    write_row(
                        scope="language_group",
                        language=language,
                        group=group,
                        values=language_summary["groups"][group],
                    )

    def close(self) -> None:
        self.speaker.close()
        self.group.close()
        self.audio_tag.close()
        self.interaction_event.close()
        self.interaction_score.close()
        self.interaction_declared.close()
        self.failure.close()
        for path, values in (
            (self.interaction_event_path, self.existing_interaction_events.values()),
            (self.interaction_score_path, self.existing_interaction_scores.values()),
            (
                self.interaction_declared_path,
                self.existing_interaction_declared.values(),
            ),
        ):
            path.write_text(
                "".join(canonical_json(row) + "\n" for row in values),
                encoding="utf-8",
            )
