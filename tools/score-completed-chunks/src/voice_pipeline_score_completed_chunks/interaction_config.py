from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

AUTOMATIC_ONLY = True


@dataclass(frozen=True, slots=True)
class InteractionConfig:
    """Frozen, serializable protocol for interaction analysis."""

    schema_version: int = 7
    vad_frame_ms: int = 30
    minimum_active_segment_ms: int = 100
    merge_inactive_gap_ms: int = 100
    minimum_cross_speaker_overlap_ms: int = 60
    clean_transition_maximum_gap_ms: int = 500
    short_response_maximum_duration_ms: int = 1_500
    floor_resumption_window_ms: int = 1_000
    turn_source_jaccard_minimum: float = 0.50
    overlap_source_jaccard_minimum: float = 0.50
    overlap_event_merge_gap_ms: int = 500
    overlap_fragment_merge_gap_ms: int = 300
    vad_minimum_dbfs: float = -50.0
    vad_maximum_dbfs: float = -25.0
    vad_noise_margin_db: float = 12.0
    nonverbal_threshold: float = 0.25
    category_minimum_support: int = 5
    bootstrap_samples: int = 10_000
    seed: int = 20_260_828

    def __post_init__(self) -> None:
        positive = (
            self.vad_frame_ms,
            self.minimum_active_segment_ms,
            self.merge_inactive_gap_ms,
            self.minimum_cross_speaker_overlap_ms,
            self.clean_transition_maximum_gap_ms,
            self.short_response_maximum_duration_ms,
            self.floor_resumption_window_ms,
            self.overlap_event_merge_gap_ms,
            self.overlap_fragment_merge_gap_ms,
            self.category_minimum_support,
            self.bootstrap_samples,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "interaction timing, support, and bootstrap values must be positive"
            )
        if not 0.0 < self.nonverbal_threshold < 1.0:
            raise ValueError("nonverbal threshold must be between zero and one")
        if not 0.0 <= self.turn_source_jaccard_minimum <= 1.0:
            raise ValueError("turn source Jaccard minimum must be between zero and one")
        if not 0.0 <= self.overlap_source_jaccard_minimum <= 1.0:
            raise ValueError(
                "overlap source Jaccard minimum must be between zero and one"
            )
        if self.vad_minimum_dbfs >= self.vad_maximum_dbfs:
            raise ValueError("VAD dBFS bounds are invalid")
    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "estimand": "event-weighted-micro-with-source-cluster-bootstrap",
            "vad_implementation": "deterministic-adaptive-frame-rms-v1",
            "primary_activity_source": "full-track-vad",
            "transcript_usage": "post-detection-attribution-only",
            "provisional_automatic_only": AUTOMATIC_ONLY,
        }

    def detector_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "vad_frame_ms": self.vad_frame_ms,
            "minimum_active_segment_ms": self.minimum_active_segment_ms,
            "merge_inactive_gap_ms": self.merge_inactive_gap_ms,
            "minimum_cross_speaker_overlap_ms": (self.minimum_cross_speaker_overlap_ms),
            "clean_transition_maximum_gap_ms": (self.clean_transition_maximum_gap_ms),
            "short_response_maximum_duration_ms": (
                self.short_response_maximum_duration_ms
            ),
            "floor_resumption_window_ms": self.floor_resumption_window_ms,
            "turn_source_jaccard_minimum": self.turn_source_jaccard_minimum,
            "overlap_source_jaccard_minimum": (
                self.overlap_source_jaccard_minimum
            ),
            "overlap_event_merge_gap_ms": self.overlap_event_merge_gap_ms,
            "overlap_fragment_merge_gap_ms": self.overlap_fragment_merge_gap_ms,
            "vad_minimum_dbfs": self.vad_minimum_dbfs,
            "vad_maximum_dbfs": self.vad_maximum_dbfs,
            "vad_noise_margin_db": self.vad_noise_margin_db,
            "nonverbal_threshold": self.nonverbal_threshold,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.detector_manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
