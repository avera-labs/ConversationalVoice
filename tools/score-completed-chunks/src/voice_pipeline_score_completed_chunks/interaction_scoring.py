from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .audio import Audio, read_wav, slice_audio
from .contracts import GROUPS, GroupDescriptor, parse_group, validate_transcript
from .errors import ScoringError, error_code
from .interaction_config import AUTOMATIC_ONLY, InteractionConfig
from .interaction_events import (
    OVERLAP_CATEGORIES,
    StageAnalysis,
    Turn,
    TransitionEvent,
    build_stage_analysis,
    effective_conversation_bounds,
    stage_metrics,
)
from .interaction_transcript import (
    InteractionUtterance,
    attach_source_indices,
    normalize_interaction_utterances,
    source_with_indices,
)
from .nonverbal import NonverbalDetector
from .repository import CompletedChunk
from .storage import ObjectStorage
from .vad import EnergyVad, Interval


@dataclass(frozen=True, slots=True)
class LoadedInteractionStage:
    descriptor: GroupDescriptor
    transcript: dict
    utterances: tuple[InteractionUtterance, ...]
    audio: tuple[Audio, Audio]
    analysis: StageAnalysis


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else None
    recall = true_positive / recall_denominator if recall_denominator else None
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else None
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _turn_match_details(
    source_turn: Turn,
    reconstruction_turn: Turn,
    *,
    config: InteractionConfig,
) -> tuple[float, float, float] | None:
    if source_turn.speaker_id != reconstruction_turn.speaker_id:
        return None
    source_ids = set(source_turn.source_indices)
    reconstruction_ids = set(reconstruction_turn.source_indices)
    intersection = source_ids.intersection(reconstruction_ids)
    if not intersection:
        return None
    jaccard = len(intersection) / len(source_ids.union(reconstruction_ids))
    if jaccard < config.turn_source_jaccard_minimum:
        return None
    return jaccard, 0.0, 0.0


def _monotonic_turn_matches(
    candidates: list[list[tuple[float, float, float] | None]],
) -> tuple[tuple[int, int], ...]:
    """Return the best one-to-one, order-preserving binary event matches."""

    empty = ((0, 0.0, 0.0, 0.0), tuple())
    states = [
        [empty for _ in range(len(candidates[0]) + 1)]
        for _ in range(len(candidates) + 1)
    ] if candidates and candidates[0] else [[empty]]
    for source_index in range(1, len(candidates) + 1):
        for reconstruction_index in range(1, len(candidates[0]) + 1):
            options = [
                states[source_index - 1][reconstruction_index],
                states[source_index][reconstruction_index - 1],
            ]
            details = candidates[source_index - 1][reconstruction_index - 1]
            if details is not None:
                previous_score, previous_pairs = states[source_index - 1][
                    reconstruction_index - 1
                ]
                jaccard, interval_iou, onset_error = details
                options.append(
                    (
                        (
                            previous_score[0] + 1,
                            previous_score[1] + jaccard,
                            previous_score[2] + interval_iou,
                            previous_score[3] - onset_error,
                        ),
                        previous_pairs
                        + ((source_index - 1, reconstruction_index - 1),),
                    )
                )
            states[source_index][reconstruction_index] = max(
                options,
                key=lambda value: (
                    value[0],
                    tuple((-left, -right) for left, right in value[1]),
                ),
            )
    return states[-1][-1][1]


def _maximum_turn_match_count(
    candidates: list[list[tuple[float, float, float] | None]],
) -> int:
    """Return unconstrained match cardinality for order-error diagnostics."""

    matched_source_by_reconstruction: dict[int, int] = {}

    def augment(source_index: int, seen: set[int]) -> bool:
        ordered = sorted(
            (
                (reconstruction_index, details)
                for reconstruction_index, details in enumerate(candidates[source_index])
                if details is not None
            ),
            key=lambda value: (
                -value[1][0],
                -value[1][1],
                value[1][2],
                value[0],
            ),
        )
        for reconstruction_index, _details in ordered:
            if reconstruction_index in seen:
                continue
            seen.add(reconstruction_index)
            previous_source = matched_source_by_reconstruction.get(
                reconstruction_index
            )
            if previous_source is None or augment(previous_source, seen):
                matched_source_by_reconstruction[reconstruction_index] = source_index
                return True
        return False

    for source_index in range(len(candidates)):
        augment(source_index, set())
    return len(matched_source_by_reconstruction)


def _turn_event_metrics(
    source: StageAnalysis,
    reconstruction: StageAnalysis,
    config: InteractionConfig,
) -> dict[str, object]:
    source_turns = [turn for turn in source.turns if turn.source_indices]
    reconstruction_turns = [
        turn for turn in reconstruction.turns if turn.source_indices
    ]
    source_owner = {
        source_index: turn.turn_id
        for turn in source_turns
        for source_index in turn.source_indices
    }
    mapped: dict[int, list] = {}
    split = 0
    merged_source_turns: set[int] = set()
    for source_turn in source_turns:
        source_ids = set(source_turn.source_indices)
        candidates = [
            turn
            for turn in reconstruction_turns
            if source_ids.intersection(turn.source_indices)
        ]
        if candidates:
            mapped[source_turn.turn_id] = candidates
        if len(candidates) > 1:
            split += 1
            continue
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        owners = {
            source_owner[index]
            for index in candidate.source_indices
            if index in source_owner
        }
        if len(owners) > 1:
            merged_source_turns.update(owners)
            continue
    candidates = [
        [
            _turn_match_details(
                source_turn,
                reconstruction_turn,
                config=config,
            )
            for reconstruction_turn in reconstruction_turns
        ]
        for source_turn in source_turns
    ]
    valid = _monotonic_turn_matches(candidates)
    preserved = len(valid)
    order_errors = max(0, _maximum_turn_match_count(candidates) - preserved)
    prf = _prf(
        preserved,
        len(reconstruction_turns) - preserved,
        len(source_turns) - preserved,
    )
    return {
        "turn_event": prf,
        "turn_event_tp": prf["true_positive"],
        "turn_event_fp": prf["false_positive"],
        "turn_event_fn": prf["false_negative"],
        "turn_event_precision": prf["precision"],
        "turn_event_recall": prf["recall"],
        "turn_event_f1": prf["f1"],
        "turn_preserved_count": preserved,
        "turn_source_count": len(source_turns),
        "turn_mapped_count": len(mapped),
        "turn_split_count": split,
        "turn_merge_count": len(merged_source_turns),
        "speaker_order_error_count": order_errors,
        "turn_preservation": prf["recall"],
        "turn_matching_coverage": prf["recall"],
    }


def _event_match(
    source: TransitionEvent,
    candidates: tuple[TransitionEvent, ...],
) -> TransitionEvent | None:
    source_current = set(source.current_source_indices)
    source_anchor = set(source.anchor_source_indices)
    matched = [
        candidate
        for candidate in candidates
        if source_current.intersection(candidate.current_source_indices)
        and source_anchor.intersection(candidate.anchor_source_indices)
        and candidate.speaker_id == source.speaker_id
    ]
    if not matched:
        return None
    return max(
        matched,
        key=lambda candidate: (
            len(source_current.intersection(candidate.current_source_indices))
            + len(source_anchor.intersection(candidate.anchor_source_indices)),
            -abs(candidate.overlap_duration_ms - source.overlap_duration_ms),
        ),
    )


def _mapped_turn(source_indices: tuple[int, ...], analysis: StageAnalysis):
    wanted = set(source_indices)
    matches = [
        turn for turn in analysis.turns if wanted.intersection(turn.source_indices)
    ]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True, slots=True)
class _OverlapEpisode:
    start_ms: int
    end_ms: int
    source_sets: tuple[frozenset[int], frozenset[int]]


def _overlap_source_sets(analysis: StageAnalysis, start_ms: int, end_ms: int):
    per_speaker: list[set[int]] = [set(), set()]
    for utterance in analysis.acoustic_utterances:
        if (
            utterance.source_index is not None
            and utterance.start_ms < end_ms
            and start_ms < utterance.end_ms
        ):
            per_speaker[utterance.speaker_id].add(utterance.source_index)
    return frozenset(per_speaker[0]), frozenset(per_speaker[1])


def _overlap_episodes(
    analysis: StageAnalysis, config: InteractionConfig
) -> list[_OverlapEpisode]:
    episodes: list[_OverlapEpisode] = []
    for interval in analysis.overlap_intervals:
        source_sets = _overlap_source_sets(
            analysis, interval.start_ms, interval.end_ms
        )
        if not all(source_sets):
            continue
        current = _OverlapEpisode(interval.start_ms, interval.end_ms, source_sets)
        if (
            episodes
            and episodes[-1].source_sets == current.source_sets
            and current.start_ms - episodes[-1].end_ms
            <= config.overlap_fragment_merge_gap_ms
        ):
            previous = episodes.pop()
            episodes.append(
                _OverlapEpisode(
                    previous.start_ms,
                    max(previous.end_ms, current.end_ms),
                    previous.source_sets,
                )
            )
        else:
            episodes.append(current)
    return episodes


def _source_jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left.union(right)
    return len(left.intersection(right)) / len(union) if union else 0.0


def _maximum_quality_matches(
    candidates: list[list[float | None]],
) -> tuple[tuple[int, int], ...]:
    """Return maximum-cardinality one-to-one matches with quality tie-breaking."""

    matched_source_by_reconstruction: dict[int, int] = {}

    def augment(source_index: int, seen: set[int]) -> bool:
        ordered = sorted(
            (
                (reconstruction_index, quality)
                for reconstruction_index, quality in enumerate(candidates[source_index])
                if quality is not None
            ),
            key=lambda value: (-value[1], value[0]),
        )
        for reconstruction_index, _quality in ordered:
            if reconstruction_index in seen:
                continue
            seen.add(reconstruction_index)
            previous_source = matched_source_by_reconstruction.get(
                reconstruction_index
            )
            if previous_source is None or augment(previous_source, seen):
                matched_source_by_reconstruction[reconstruction_index] = source_index
                return True
        return False

    for source_index in range(len(candidates)):
        augment(source_index, set())
    return tuple(
        sorted(
            (source_index, reconstruction_index)
            for reconstruction_index, source_index in (
                matched_source_by_reconstruction.items()
            )
        )
    )


def _overlap_event_metrics(
    source: StageAnalysis,
    reconstruction: StageAnalysis,
    config: InteractionConfig,
) -> dict[str, object]:
    source_episodes = _overlap_episodes(source, config)
    reconstruction_episodes = _overlap_episodes(reconstruction, config)
    candidates: list[list[float | None]] = []
    for source_episode in source_episodes:
        row: list[float | None] = []
        for reconstruction_episode in reconstruction_episodes:
            similarities = tuple(
                _source_jaccard(left, right)
                for left, right in zip(
                    source_episode.source_sets,
                    reconstruction_episode.source_sets,
                    strict=True,
                )
            )
            row.append(
                sum(similarities) / len(similarities)
                if all(
                    value >= config.overlap_source_jaccard_minimum
                    for value in similarities
                )
                else None
            )
        candidates.append(row)
    matches = _maximum_quality_matches(candidates)
    matched_source = {source_index for source_index, _ in matches}
    matched_reconstruction = {
        reconstruction_index for _, reconstruction_index in matches
    }
    prf = _prf(
        len(matched_source),
        len(reconstruction_episodes) - len(matched_reconstruction),
        len(source_episodes) - len(matched_source),
    )
    return {
        "overlap_event": prf,
        "overlap_event_tp": prf["true_positive"],
        "overlap_event_fp": prf["false_positive"],
        "overlap_event_fn": prf["false_negative"],
        "overlap_event_precision": prf["precision"],
        "overlap_event_recall": prf["recall"],
        "overlap_event_f1": prf["f1"],
    }


def _backchannel_event_metrics(
    source: StageAnalysis,
    reconstruction: StageAnalysis,
    config: InteractionConfig,
) -> dict[str, object]:
    source_events = [
        event for event in source.transitions if event.category == "backchannel"
    ]
    source_backchannel_indices = {
        source_index
        for event in source_events
        for source_index in event.current_source_indices
    }
    reconstruction_events = [
        event
        for event in reconstruction.transitions
        if event.category == "backchannel"
        or (
            event.category != "interruption"
            and source_backchannel_indices.intersection(event.current_source_indices)
        )
    ]
    candidates: list[list[float | None]] = []
    for event in source_events:
        row: list[float | None] = []
        source_anchor = frozenset(event.anchor_source_indices)
        source_current = frozenset(event.current_source_indices)
        for candidate in reconstruction_events:
            if candidate.speaker_id != event.speaker_id:
                row.append(None)
                continue
            anchor_similarity = _source_jaccard(
                source_anchor, frozenset(candidate.anchor_source_indices)
            )
            current_similarity = _source_jaccard(
                source_current, frozenset(candidate.current_source_indices)
            )
            row.append(
                (anchor_similarity + current_similarity) / 2
                if anchor_similarity > 0
                and current_similarity >= config.turn_source_jaccard_minimum
                else None
            )
        candidates.append(row)
    matches = _maximum_quality_matches(candidates)
    matched_source = {source_index for source_index, _ in matches}
    matched_reconstruction = {
        reconstruction_index for _, reconstruction_index in matches
    }
    prf = _prf(
        len(matched_source),
        len(reconstruction_events) - len(matched_reconstruction),
        len(source_events) - len(matched_source),
    )
    return {
        "backchannel_event": prf,
        "backchannel_event_tp": prf["true_positive"],
        "backchannel_event_fp": prf["false_positive"],
        "backchannel_event_fn": prf["false_negative"],
        "backchannel_event_precision": prf["precision"],
        "backchannel_event_recall": prf["recall"],
        "backchannel_event_f1": prf["f1"],
    }


def _reconstruction_metrics(
    source: StageAnalysis,
    reconstruction: StageAnalysis,
    config: InteractionConfig,
) -> dict[str, object]:
    source_start, source_end = effective_conversation_bounds(source)
    reconstruction_start, reconstruction_end = effective_conversation_bounds(
        reconstruction
    )
    source_duration = source_end - source_start
    reconstruction_duration = reconstruction_end - reconstruction_start
    alpha = reconstruction_duration / source_duration
    result = _turn_event_metrics(source, reconstruction, config)
    result.update(_overlap_event_metrics(source, reconstruction, config))
    result.update(_backchannel_event_metrics(source, reconstruction, config))
    result.update(
        {
            "duration_ratio": alpha,
            "duration_log_error": abs(math.log(alpha)),
            "source_effective_duration_ms": source_duration,
            "reconstruction_effective_duration_ms": reconstruction_duration,
        }
    )
    gap_errors: list[float] = []
    signed_gap_errors: list[float] = []
    gap_buckets: dict[str, list[float]] = {
        "0_200": [],
        "200_500": [],
        "500_1000": [],
        "over_1000": [],
    }
    for left, right in zip(source.turns, source.turns[1:]):
        if left.speaker_id == right.speaker_id or right.start_ms < left.end_ms:
            continue
        mapped_left = _mapped_turn(left.source_indices, reconstruction)
        mapped_right = _mapped_turn(right.source_indices, reconstruction)
        if mapped_left is None or mapped_right is None:
            continue
        source_gap = right.start_ms - left.end_ms
        reconstruction_gap = mapped_right.start_ms - mapped_left.end_ms
        source_local_duration = left.duration_ms + right.duration_ms
        reconstruction_local_duration = (
            mapped_left.duration_ms + mapped_right.duration_ms
        )
        if source_local_duration <= 0 or reconstruction_local_duration <= 0:
            continue
        local_alpha = reconstruction_local_duration / source_local_duration
        signed = reconstruction_gap / local_alpha - source_gap
        signed_gap_errors.append(float(signed))
        gap_errors.append(float(abs(signed)))
        bucket = (
            "0_200"
            if source_gap < 200
            else "200_500"
            if source_gap < 500
            else "500_1000"
            if source_gap < 1000
            else "over_1000"
        )
        gap_buckets[bucket].append(float(abs(signed)))

    source_overlaps = [
        event
        for event in source.transitions
        if event.category in OVERLAP_CATEGORIES
        and event.overlap_duration_ms >= config.minimum_cross_speaker_overlap_ms
    ]
    preserved_overlaps = int(result.get("overlap_event_tp", 0))
    overlap_relative_errors: list[float] = []
    overlap_absolute_errors: list[float] = []
    source_backchannels = [
        event for event in source.transitions if event.category == "backchannel"
    ]
    preserved_backchannels = int(result.get("backchannel_event_tp", 0))
    overlap_onset_errors: list[float] = []
    mapped_source_overlap_duration = 0
    mapped_reconstruction_overlap_duration = 0
    for event in source_overlaps:
        candidate = _event_match(event, reconstruction.transitions)
        if candidate is None:
            continue
        if candidate.overlap_duration_ms >= config.minimum_cross_speaker_overlap_ms:
            preserved_overlaps += 1
        absolute = abs(candidate.overlap_duration_ms - event.overlap_duration_ms)
        mapped_source_overlap_duration += event.overlap_duration_ms
        mapped_reconstruction_overlap_duration += candidate.overlap_duration_ms
        overlap_absolute_errors.append(float(absolute))
        overlap_relative_errors.append(absolute / event.overlap_duration_ms)
    for event in source_backchannels:
        candidate = _event_match(event, reconstruction.transitions)
        if candidate is None:
            continue
        source_anchor = source.turns[event.anchor_turn_id]
        reconstruction_anchor = _mapped_turn(
            event.anchor_source_indices, reconstruction
        )
        if reconstruction_anchor is None:
            continue
        source_position = (event.start_ms - source_anchor.start_ms) / max(
            1, source_anchor.duration_ms
        )
        reconstruction_position = (
            candidate.start_ms - reconstruction_anchor.start_ms
        ) / max(1, reconstruction_anchor.duration_ms)
        position_error = (
            abs(source_position - reconstruction_position) * source_anchor.duration_ms
        )
        overlap_onset_errors.append(position_error)

    overlap_denominator = preserved_overlaps + int(
        result.get("overlap_event_fn", 0)
    )
    backchannel_denominator = preserved_backchannels + int(
        result.get("backchannel_event_fn", 0)
    )
    result.update(
        {
            "gap_errors_ms": gap_errors,
            "signed_gap_errors_ms": signed_gap_errors,
            "gap_error_median_ms": _median(gap_errors),
            "gap_error_mae_ms": _mean(gap_errors),
            "gap_error_p90_ms": _percentile(gap_errors, 90),
            "gap_error_signed_bias_ms": _mean(signed_gap_errors),
            "gap_error_buckets": {
                bucket: {
                    "count": len(values),
                    "median_ms": _median(values),
                    "mae_ms": _mean(values),
                    "p90_ms": _percentile(values, 90),
                }
                for bucket, values in gap_buckets.items()
            },
            "overlap_source_count": overlap_denominator,
            "overlap_preserved_count": preserved_overlaps,
            "overlap_preservation": (
                preserved_overlaps / overlap_denominator
                if overlap_denominator
                else None
            ),
            "overlap_relative_errors": overlap_relative_errors,
            "overlap_absolute_errors_ms": overlap_absolute_errors,
            "overlap_error_median": _median(overlap_relative_errors),
            "overlap_error_mean": _mean(overlap_relative_errors),
            "overlap_error_p90": _percentile(overlap_relative_errors, 90),
            "overlap_absolute_error_p90_ms": _percentile(overlap_absolute_errors, 90),
            "overlap_total_error": (
                abs(
                    mapped_source_overlap_duration
                    - mapped_reconstruction_overlap_duration
                )
                / mapped_source_overlap_duration
                if mapped_source_overlap_duration
                else None
            ),
            "overlap_mapping_coverage": (
                preserved_overlaps / overlap_denominator
                if overlap_denominator
                else None
            ),
            "overlap_onset_errors_ms": overlap_onset_errors,
            "backchannel_source_count": backchannel_denominator,
            "backchannel_preserved_count": preserved_backchannels,
            "backchannel_preservation": (
                preserved_backchannels / backchannel_denominator
                if backchannel_denominator
                else None
            ),
            "backchannel_false_positive_count": int(
                result.get("backchannel_event_fp", 0)
            ),
            "source_event_mapping_count": sum(
                _event_match(event, reconstruction.transitions) is not None
                for event in source.transitions
            ),
            "source_event_count": len(source.transitions),
            "source_event_mapping_coverage": (
                sum(
                    _event_match(event, reconstruction.transitions) is not None
                    for event in source.transitions
                )
                / len(source.transitions)
                if source.transitions
                else None
            ),
        }
    )
    return result


class InteractionScoreEngine:
    def __init__(
        self,
        *,
        storage: ObjectStorage,
        config: InteractionConfig,
        vad: EnergyVad,
        nonverbal: NonverbalDetector,
    ) -> None:
        self.storage = storage
        self.config = config
        self.vad = vad
        self.nonverbal = nonverbal
        self.nonverbal_available = (
            nonverbal.manifest().get("implementation") != "disabled"
        )
        encoded = json.dumps(
            {
                "config": config.detector_manifest(),
                "vad": vad.manifest(),
                "nonverbal": nonverbal.manifest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.detector_fingerprint = hashlib.sha256(encoded).hexdigest()

    def manifest(self) -> dict[str, object]:
        return {
            "detector_fingerprint": self.detector_fingerprint,
            "config": self.config.manifest(),
            "vad": self.vad.manifest(),
            "nonverbal": self.nonverbal.manifest(),
        }

    def _resume_key(self, descriptors: dict[str, GroupDescriptor]) -> str:
        identities = [self.detector_fingerprint]
        for group in GROUPS:
            if group not in descriptors:
                continue
            descriptor = descriptors[group]
            identities.append(descriptor.transcript.sha256)
            identities.extend(track.artifact.sha256 for track in descriptor.tracks)
        return hashlib.sha256(":".join(identities).encode("ascii")).hexdigest()

    def _load_stage(self, descriptor: GroupDescriptor) -> LoadedInteractionStage:
        payload = self.storage.download(descriptor.transcript)
        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScoringError("invalid_transcript_json") from exc
        transcript = validate_transcript(raw, group=descriptor)
        utterances = normalize_interaction_utterances(transcript, group=descriptor)
        audios: list[Audio] = []
        activities: list[tuple[Interval, ...]] = []
        for track in descriptor.tracks:
            audio = read_wav(
                self.storage.download(track.artifact),
                expected_rate=track.sample_rate_hz,
            )
            if audio.duration_ms != descriptor.duration_ms:
                raise ScoringError("track_duration_mismatch")
            audios.append(audio)
            detected = self.vad.intervals(audio)
            if not detected:
                raise ScoringError("interaction_empty_vad")
            activities.append(detected)
        analysis = build_stage_analysis(
            utterances=utterances,
            activities=(activities[0], activities[1]),
            duration_ms=descriptor.duration_ms,
            config=self.config,
        )
        return LoadedInteractionStage(
            descriptor=descriptor,
            transcript=transcript,
            utterances=utterances,
            audio=(audios[0], audios[1]),
            analysis=analysis,
        )

    def _nonverbal_events(
        self, stage: LoadedInteractionStage
    ) -> tuple[list[dict], set[int]]:
        if not self.nonverbal_available:
            return [], set()
        events: list[dict] = []
        observed_utterances: set[int] = set()
        for utterance in stage.utterances:
            clip = slice_audio(
                stage.audio[utterance.speaker_id],
                start_ms=utterance.start_ms,
                end_ms=utterance.end_ms,
            )
            for observation in self.nonverbal.detect(clip):
                observed_utterances.add(utterance.utterance_index)
                events.append(
                    {
                        "event_id": f"nonverbal-{utterance.utterance_index}-{observation.family}",
                        "event_kind": "nonverbal",
                        "event_type": observation.family,
                        "speaker_id": utterance.speaker_id,
                        "other_speaker_id": None,
                        "acoustic_start_ms": utterance.start_ms,
                        "acoustic_end_ms": utterance.end_ms,
                        "duration_ms": utterance.duration_ms,
                        "overlap_duration_ms": 0,
                        "transcript_utterance_indices": [utterance.utterance_index],
                        "detector_confidence": observation.confidence,
                        "detector_label": observation.label,
                        "rule_trace": ["ast_label_family_mapping"],
                    }
                )
        return events, observed_utterances

    def _stage_rows(
        self,
        *,
        chunk: CompletedChunk,
        group: str,
        stage: LoadedInteractionStage,
        resume_key: str,
        audio_tag_rows: list[dict],
    ) -> tuple[list[dict], dict, list[dict]]:
        events = [
            {
                "event_id": event.event_id,
                "event_kind": "transition",
                "event_type": event.category,
                "speaker_id": event.speaker_id,
                "other_speaker_id": event.other_speaker_id,
                "acoustic_start_ms": event.start_ms,
                "acoustic_end_ms": event.end_ms,
                "duration_ms": event.end_ms - event.start_ms,
                "overlap_duration_ms": event.overlap_duration_ms,
                "gap_ms": event.gap_ms,
                "transcript_utterance_indices": list(event.current_utterance_indices),
                "anchor_source_indices": list(event.anchor_source_indices),
                "current_source_indices": list(event.current_source_indices),
                "source_event_id": None,
                "detector_confidence": None,
                "rule_trace": list(event.rule_trace),
            }
            for event in stage.analysis.transitions
        ]
        nonverbal_events, observed_nonverbal = self._nonverbal_events(stage)
        events.extend(nonverbal_events)
        common = {
            "schema_version": 1,
            "chunk_id": str(chunk.chunk_id),
            "source_cluster_id": str(chunk.source_cluster_id or chunk.chunk_id),
            "language": chunk.language,
            "group": group,
            "track_sha256": [
                track.artifact.sha256 for track in stage.descriptor.tracks
            ],
            "transcript_sha256": stage.descriptor.transcript.sha256,
            "detector_fingerprint": self.detector_fingerprint,
            "resume_key": resume_key,
        }
        event_rows = [{**common, **event} for event in events]
        metrics = stage_metrics(stage.analysis)
        eligible_utterances = len(stage.utterances)
        nonverbal_count = len(observed_nonverbal)
        score = {
            **common,
            **metrics,
            "eligible_utterance_count": eligible_utterances,
            "observed_nonverbal_utterance_count": (
                nonverbal_count if self.nonverbal_available else None
            ),
            "nonverbal_utterance_rate": (
                nonverbal_count / eligible_utterances
                if self.nonverbal_available and eligible_utterances
                else None
            ),
            "observed_nonverbal_event_count": (
                len(nonverbal_events) if self.nonverbal_available else None
            ),
            "nonverbal_density_per_active_speech_minute": (
                len(nonverbal_events) / metrics["active_speech_minutes"]
                if self.nonverbal_available and metrics["active_speech_minutes"]
                else None
            ),
            "nonverbal_available": self.nonverbal_available,
            "support_set": sorted(
                set(metrics["observed_transition_categories"])
                | {event["event_type"] for event in nonverbal_events}
            ),
            "event_count": len(event_rows),
            "declared_row_count": len(stage.utterances) if group != "separation" else 0,
            "status": "success",
            "error_code": None,
            "provisional_automatic_only": AUTOMATIC_ONLY,
        }
        tag_by_index = {
            int(row["transcript_index"]): row
            for row in audio_tag_rows
            if row.get("group") == group
            and isinstance(row.get("transcript_index"), int)
        }
        transitions_by_utterance: dict[int, list[TransitionEvent]] = {}
        for event in stage.analysis.transitions:
            for index in event.current_utterance_indices:
                transitions_by_utterance.setdefault(index, []).append(event)
        acoustic_by_index = {
            item.utterance_index: item for item in stage.analysis.acoustic_utterances
        }
        declared_rows: list[dict] = []
        if group != "separation":
            for utterance in stage.utterances:
                observed_transitions = transitions_by_utterance.get(
                    utterance.utterance_index, []
                )
                acoustic = acoustic_by_index.get(utterance.utterance_index)
                tag_row = tag_by_index.get(utterance.utterance_index)
                declared_rows.append(
                    {
                        **common,
                        "utterance_index": utterance.utterance_index,
                        "speaker_id": utterance.speaker_id,
                        "declared_type": utterance.declared_type,
                        "declared_placement": utterance.declared_placement,
                        "declared_relation": utterance.declared_relation,
                        "declared_overlap": utterance.declared_placement
                        == "overlap_previous",
                        "observed_overlap": any(
                            event.category in OVERLAP_CATEGORIES
                            for event in observed_transitions
                        ),
                        "declared_backchannel": utterance.declared_type
                        == "backchannel",
                        "observed_backchannel": any(
                            event.category == "backchannel"
                            for event in observed_transitions
                        ),
                        "declared_paralinguistic": utterance.declared_type
                        == "paralinguistic",
                        "observed_nonverbal": (
                            utterance.utterance_index in observed_nonverbal
                            if self.nonverbal_available
                            else None
                        ),
                        "planned_start_ms": utterance.start_ms,
                        "planned_end_ms": utterance.end_ms,
                        "observed_start_ms": acoustic.start_ms if acoustic else None,
                        "observed_end_ms": acoustic.end_ms if acoustic else None,
                        "start_boundary_absolute_error_ms": (
                            abs(acoustic.start_ms - utterance.start_ms)
                            if acoustic
                            else None
                        ),
                        "end_boundary_absolute_error_ms": (
                            abs(acoustic.end_ms - utterance.end_ms)
                            if acoustic
                            else None
                        ),
                        "audio_tag_score": tag_row.get("score") if tag_row else None,
                        "audio_tag_status": tag_row.get("status") if tag_row else None,
                        "status": "success",
                    }
                )
        return event_rows, score, declared_rows

    def score_chunk(
        self,
        chunk: CompletedChunk,
        *,
        audio_tag_rows: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        descriptors: dict[str, GroupDescriptor] = {}
        failures: list[dict] = []
        score_rows: list[dict] = []
        for group in GROUPS:
            try:
                descriptors[group] = parse_group(chunk.final_results, group)
            except Exception as exc:
                failures.append(
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "scope": f"interaction:{group}:contract",
                        "error_code": error_code(exc),
                    }
                )
                score_rows.append(
                    {
                        "schema_version": 1,
                        "chunk_id": str(chunk.chunk_id),
                        "source_cluster_id": str(
                            chunk.source_cluster_id or chunk.chunk_id
                        ),
                        "language": chunk.language,
                        "group": group,
                        "status": "failed",
                        "error_code": error_code(exc),
                        "detector_fingerprint": self.detector_fingerprint,
                        "resume_key": None,
                        "provisional_automatic_only": AUTOMATIC_ONLY,
                    }
                )
        resume_key = self._resume_key(descriptors)
        loaded: dict[str, LoadedInteractionStage] = {}
        event_rows: list[dict] = []
        declared_rows: list[dict] = []
        for group, descriptor in descriptors.items():
            try:
                loaded[group] = self._load_stage(descriptor)
            except Exception as exc:
                failures.append(
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "scope": f"interaction:{group}",
                        "error_code": error_code(exc),
                    }
                )
                score_rows.append(
                    {
                        "schema_version": 1,
                        "chunk_id": str(chunk.chunk_id),
                        "source_cluster_id": str(
                            chunk.source_cluster_id or chunk.chunk_id
                        ),
                        "language": chunk.language,
                        "group": group,
                        "status": "failed",
                        "error_code": error_code(exc),
                        "detector_fingerprint": self.detector_fingerprint,
                        "resume_key": resume_key,
                        "provisional_automatic_only": AUTOMATIC_ONLY,
                    }
                )
        if "separation" in loaded:
            source = loaded["separation"]
            source_utterances = source_with_indices(source.utterances)
            loaded["separation"] = LoadedInteractionStage(
                descriptor=source.descriptor,
                transcript=source.transcript,
                utterances=source_utterances,
                audio=source.audio,
                analysis=build_stage_analysis(
                    utterances=source_utterances,
                    activities=source.analysis.activities,
                    duration_ms=source.descriptor.duration_ms,
                    config=self.config,
                ),
            )
        if "separation" in loaded and "reconstruction" in loaded:
            reconstruction = loaded["reconstruction"]
            try:
                mapped = attach_source_indices(
                    loaded["separation"].utterances,
                    reconstruction.utterances,
                    reconstruction.transcript,
                )
                loaded["reconstruction"] = LoadedInteractionStage(
                    descriptor=reconstruction.descriptor,
                    transcript=reconstruction.transcript,
                    utterances=mapped,
                    audio=reconstruction.audio,
                    analysis=build_stage_analysis(
                        utterances=mapped,
                        activities=reconstruction.analysis.activities,
                        duration_ms=reconstruction.descriptor.duration_ms,
                        config=self.config,
                    ),
                )
            except Exception as exc:
                failures.append(
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "scope": "interaction:reconstruction:mapping",
                        "error_code": error_code(exc),
                    }
                )
                loaded.pop("reconstruction")
                score_rows.append(
                    {
                        "schema_version": 1,
                        "chunk_id": str(chunk.chunk_id),
                        "source_cluster_id": str(
                            chunk.source_cluster_id or chunk.chunk_id
                        ),
                        "language": chunk.language,
                        "group": "reconstruction",
                        "status": "failed",
                        "error_code": error_code(exc),
                        "detector_fingerprint": self.detector_fingerprint,
                        "resume_key": resume_key,
                        "provisional_automatic_only": AUTOMATIC_ONLY,
                    }
                )
        current_audio_tags = audio_tag_rows or []
        for group, stage in loaded.items():
            try:
                rows, score, declared = self._stage_rows(
                    chunk=chunk,
                    group=group,
                    stage=stage,
                    resume_key=resume_key,
                    audio_tag_rows=current_audio_tags,
                )
                event_rows.extend(rows)
                declared_rows.extend(declared)
                score_rows.append(score)
            except Exception as exc:
                failures.append(
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "scope": f"interaction:{group}:events",
                        "error_code": error_code(exc),
                    }
                )
                score_rows.append(
                    {
                        "schema_version": 1,
                        "chunk_id": str(chunk.chunk_id),
                        "source_cluster_id": str(
                            chunk.source_cluster_id or chunk.chunk_id
                        ),
                        "language": chunk.language,
                        "group": group,
                        "status": "failed",
                        "error_code": error_code(exc),
                        "detector_fingerprint": self.detector_fingerprint,
                        "resume_key": resume_key,
                        "provisional_automatic_only": AUTOMATIC_ONLY,
                    }
                )
        if "separation" in loaded and "reconstruction" in loaded:
            primary = _reconstruction_metrics(
                loaded["separation"].analysis,
                loaded["reconstruction"].analysis,
                self.config,
            )
            for row in score_rows:
                if (
                    row.get("group") == "reconstruction"
                    and row.get("status") == "success"
                ):
                    row.update(primary)
                    break
            source_by_reconstruction_event: dict[str, str] = {}
            for source_event in loaded["separation"].analysis.transitions:
                candidate = _event_match(
                    source_event, loaded["reconstruction"].analysis.transitions
                )
                if candidate is not None:
                    source_by_reconstruction_event[candidate.event_id] = (
                        source_event.event_id
                    )
            for row in event_rows:
                if (
                    row.get("group") == "reconstruction"
                    and row.get("event_kind") == "transition"
                ):
                    row["source_event_id"] = source_by_reconstruction_event.get(
                        str(row["event_id"])
                    )
                    row["match_outcome"] = (
                        "true_positive" if row["source_event_id"] else "false_positive"
                    )
            matched_source_ids = set(source_by_reconstruction_event.values())
            for row in event_rows:
                if (
                    row.get("group") == "separation"
                    and row.get("event_kind") == "transition"
                ):
                    row["match_outcome"] = (
                        "true_positive"
                        if row.get("event_id") in matched_source_ids
                        else "false_negative"
                    )
        successful_by_group = {
            str(row.get("group")): row
            for row in score_rows
            if row.get("status") == "success"
        }
        separation_score = successful_by_group.get("separation")
        reconstruction_score = successful_by_group.get("reconstruction")
        if separation_score is not None and reconstruction_score is not None:
            separation_duration = separation_score.get("conversation_duration_ms")
            reconstruction_duration = reconstruction_score.get(
                "conversation_duration_ms"
            )
            if isinstance(separation_duration, int) and separation_duration > 0:
                if (
                    isinstance(reconstruction_duration, int)
                    and reconstruction_duration > 0
                ):
                    reconstruction_score["duration_ratio"] = (
                        reconstruction_duration / separation_duration
                    )
            expansion_score = successful_by_group.get("expansion")
            if (
                expansion_score is not None
                and isinstance(reconstruction_duration, int)
                and reconstruction_duration > 0
            ):
                expansion_duration = expansion_score.get("conversation_duration_ms")
                if isinstance(expansion_duration, int) and expansion_duration > 0:
                    expansion_score["expansion_factor"] = (
                        expansion_duration / reconstruction_duration
                    )
        return event_rows, score_rows, declared_rows, failures
