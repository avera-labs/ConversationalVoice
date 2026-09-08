from __future__ import annotations

import hashlib
import io
import json
import wave
from datetime import UTC, datetime
from uuid import UUID

import numpy as np

from voice_pipeline_score_completed_chunks.interaction_config import InteractionConfig
from voice_pipeline_score_completed_chunks.interaction_events import (
    build_stage_analysis,
)
from voice_pipeline_score_completed_chunks.interaction_scoring import (
    InteractionScoreEngine,
    _reconstruction_metrics,
    _turn_event_metrics,
)
from voice_pipeline_score_completed_chunks.interaction_transcript import (
    InteractionUtterance,
)
from voice_pipeline_score_completed_chunks.nonverbal import DisabledNonverbalDetector
from voice_pipeline_score_completed_chunks.repository import CompletedChunk
from voice_pipeline_score_completed_chunks.storage import StoredObject
from voice_pipeline_score_completed_chunks.vad import EnergyVad
from voice_pipeline_score_completed_chunks.vad import Interval


def wav_with_activity(
    sample_rate: int, duration_ms: int, intervals: list[tuple[int, int]]
) -> bytes:
    samples = np.zeros(round(duration_ms * sample_rate / 1000), dtype=np.float64)
    for start_ms, end_ms in intervals:
        start = round(start_ms * sample_rate / 1000)
        end = round(end_ms * sample_rate / 1000)
        time = np.arange(end - start) / sample_rate
        samples[start:end] = np.sin(2 * np.pi * 220 * time) * 0.25
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
    return target.getvalue()


def identity(uri: str, payload: bytes, **extra) -> dict:
    return {
        "uri": uri,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        **extra,
    }


class FakeStorage:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def download(self, artifact: StoredObject) -> bytes:
        return self.objects[artifact.uri]


def test_interaction_engine_scores_source_relative_reconstruction() -> None:
    duration = 3500
    mapping = [
        {"speaker_id": 0, "diarization_speaker_id": 10},
        {"speaker_id": 1, "diarization_speaker_id": 11},
    ]
    source_utterances = [
        {"speaker_id": 0, "start_ms": 0, "end_ms": 1000, "text": "main point"},
        {"speaker_id": 1, "start_ms": 800, "end_ms": 950, "text": "yeah"},
        {"speaker_id": 1, "start_ms": 1300, "end_ms": 2300, "text": "answer"},
        {"speaker_id": 0, "start_ms": 2600, "end_ms": 3300, "text": "follow up"},
    ]
    reconstruction_utterances = [
        {
            **item,
            "utterance_index": index,
            "source_start_ms": item["start_ms"],
            "source_end_ms": item["end_ms"],
            "type": "backchannel" if index == 1 else "dialogue",
            "placement": "overlap_previous" if index == 1 else "sequential",
            "text_with_audio_tags": item["text"],
        }
        for index, item in enumerate(source_utterances)
    ]
    expansion_utterances = [
        {
            "utterance_index": 0,
            "speaker_id": 0,
            "start_ms": 0,
            "end_ms": 900,
            "text": "continued",
            "text_with_audio_tags": "continued",
            "type": "dialogue",
            "placement": "sequential",
        },
        {
            "utterance_index": 1,
            "speaker_id": 1,
            "start_ms": 1100,
            "end_ms": 2100,
            "text": "reply",
            "text_with_audio_tags": "reply",
            "type": "dialogue",
            "placement": "sequential",
        },
    ]
    separation_transcript = json.dumps(
        {
            "language": "en",
            "timebase": "chunk",
            "speakers": [
                {
                    "output_slot": speaker,
                    "diarization_speaker_id": 10 + speaker,
                    "utterances": [
                        {
                            key: value
                            for key, value in item.items()
                            if key != "speaker_id"
                        }
                        for item in source_utterances
                        if item["speaker_id"] == speaker
                    ],
                }
                for speaker in range(2)
            ],
        }
    ).encode()
    reconstruction_transcript = json.dumps(
        {
            "language": "en",
            "timebase": "reconstruction",
            "duration_ms": duration,
            "speaker_mapping": mapping,
            "utterances": reconstruction_utterances,
        }
    ).encode()
    expansion_transcript = json.dumps(
        {
            "language": "en",
            "timebase": "dialogue_extension",
            "duration_ms": duration,
            "speaker_mapping": mapping,
            "utterances": expansion_utterances,
        }
    ).encode()
    objects = {
        "s3://bucket/separation.json": separation_transcript,
        "s3://bucket/reconstruction.json": reconstruction_transcript,
        "s3://bucket/expansion.json": expansion_transcript,
    }
    stage_intervals = {
        "separation": [[(0, 1000), (2600, 3300)], [(800, 950), (1300, 2300)]],
        "reconstruction": [
            [(0, 1000), (2600, 3300)],
            [(800, 950), (1300, 2300)],
        ],
        "expansion": [[(0, 900)], [(1100, 2100)]],
    }
    tracks: dict[str, list[dict]] = {}
    for group, intervals in stage_intervals.items():
        rate = 16000 if group == "separation" else 44100
        tracks[group] = []
        for speaker in range(2):
            payload = wav_with_activity(rate, duration, intervals[speaker])
            uri = f"s3://bucket/{group}-{speaker}.wav"
            objects[uri] = payload
            tracks[group].append(
                identity(
                    uri,
                    payload,
                    **(
                        {
                            "output_slot": speaker,
                            "diarization_speaker_id": 10 + speaker,
                            "sample_rate_hz": rate,
                            "duration_ms": duration,
                        }
                        if group == "separation"
                        else {
                            "speaker_id": speaker,
                            "diarization_speaker_id": 10 + speaker,
                            "sample_rate_hz": rate,
                            "duration_ms": duration,
                        }
                    ),
                )
            )
    results = {
        "separation": {"speaker_audio": tracks["separation"]},
        "transcription": {
            "language": "en",
            "input_speaker_audio": [
                {
                    key: value
                    for key, value in track.items()
                    if key
                    in {
                        "output_slot",
                        "diarization_speaker_id",
                        "uri",
                        "size_bytes",
                        "sha256",
                    }
                }
                for track in tracks["separation"]
            ],
            "artifacts": {
                "transcript": identity(
                    "s3://bucket/separation.json", separation_transcript
                )
            },
        },
        "reconstruction": {
            "language": "en",
            "actual_duration_ms": duration,
            "artifacts": {
                "transcript": identity(
                    "s3://bucket/reconstruction.json", reconstruction_transcript
                ),
                "speaker_audio": tracks["reconstruction"],
            },
        },
        "dialogue_extension": {
            "language": "en",
            "actual_duration_ms": duration,
            "artifacts": {
                "transcript": identity(
                    "s3://bucket/expansion.json", expansion_transcript
                ),
                "speaker_audio": tracks["expansion"],
            },
        },
    }
    chunk = CompletedChunk(
        UUID(int=1),
        "en",
        datetime.now(UTC),
        "s3://bucket/audio.wav",
        results,
        UUID(int=99),
    )
    config = InteractionConfig(bootstrap_samples=10)
    engine = InteractionScoreEngine(
        storage=FakeStorage(objects),  # type: ignore[arg-type]
        config=config,
        vad=EnergyVad(config),
        nonverbal=DisabledNonverbalDetector(),
    )
    events, scores, declared, failures = engine.score_chunk(chunk)
    assert not failures
    assert len(scores) == 3
    reconstruction = next(row for row in scores if row["group"] == "reconstruction")
    assert reconstruction["turn_preservation"] == 1.0
    assert reconstruction["turn_event_f1"] == 1.0
    assert reconstruction["overlap_event_f1"] == 1.0
    assert reconstruction["backchannel_event_f1"] == 1.0
    assert reconstruction["duration_ratio"] == 1.0
    assert reconstruction["backchannel_preservation"] == 1.0
    assert reconstruction["gap_error_median_ms"] == 0.0
    assert "speaker_activity_iou" not in reconstruction
    assert "normalized_speaker_activity_iou" not in reconstruction
    assert "activity_state_iou" not in reconstruction
    assert "normalized_activity_state_iou" not in reconstruction
    assert any(
        row["group"] == "reconstruction" and row["source_event_id"] is not None
        for row in events
        if row["event_kind"] == "transition"
    )
    assert len(declared) == len(reconstruction_utterances) + len(expansion_utterances)
    expansion = next(row for row in scores if row["group"] == "expansion")
    assert expansion["turn_rate_per_minute"] > 0
    assert expansion["expansion_factor"] == (
        expansion["conversation_duration_ms"]
        / reconstruction["conversation_duration_ms"]
    )


def _analysis(utterances, activities, duration=2500):
    return build_stage_analysis(
        utterances=tuple(utterances),
        activities=(
            tuple(Interval(*value) for value in activities[0]),
            tuple(Interval(*value) for value in activities[1]),
        ),
        duration_ms=duration,
        config=InteractionConfig(),
    )


def _utterance(index, speaker, start, end, source_index):
    return InteractionUtterance(
        index,
        speaker,
        start,
        end,
        f"utterance {index}",
        source_index=source_index,
    )


def test_turn_event_f1_counts_split_merge_and_order_errors() -> None:
    source = _analysis(
        [_utterance(0, 0, 0, 300, 0), _utterance(1, 0, 800, 1200, 1)],
        [[(0, 300), (800, 1200)], []],
    )
    split = _analysis(
        [
            _utterance(0, 0, 0, 150, 0),
            _utterance(1, 0, 400, 600, 0),
            _utterance(2, 0, 800, 1200, 1),
        ],
        [[(0, 150), (400, 600), (800, 1200)], []],
    )
    config = InteractionConfig()
    split_metrics = _turn_event_metrics(source, split, config)
    assert split_metrics["turn_event_tp"] == 2
    assert split_metrics["turn_event_fp"] == 1
    assert split_metrics["turn_event_fn"] == 0
    assert split_metrics["turn_event_f1"] == 0.8
    assert split_metrics["turn_split_count"] == 1

    merged = _analysis(
        [_utterance(0, 0, 0, 300, 0), _utterance(1, 0, 350, 750, 1)],
        [[(0, 300), (350, 750)], []],
    )
    merged_metrics = _turn_event_metrics(source, merged, config)
    assert merged_metrics["turn_event_tp"] == 1
    assert merged_metrics["turn_event_fp"] == 0
    assert merged_metrics["turn_event_fn"] == 1
    assert merged_metrics["turn_event_f1"] == 2 / 3
    assert merged_metrics["turn_merge_count"] == 2

    ordered_source = _analysis(
        [_utterance(0, 0, 0, 300, 0), _utterance(1, 1, 500, 800, 1)],
        [[(0, 300)], [(500, 800)]],
    )
    reversed_reconstruction = _analysis(
        [_utterance(0, 1, 0, 300, 1), _utterance(1, 0, 500, 800, 0)],
        [[(500, 800)], [(0, 300)]],
    )
    order_metrics = _turn_event_metrics(
        ordered_source, reversed_reconstruction, config
    )
    assert order_metrics["speaker_order_error_count"] == 1
    assert order_metrics["turn_event_tp"] == 1


def test_turn_event_matching_uses_source_overlap_without_global_time() -> None:
    config = InteractionConfig()
    source = _analysis(
        [
            _utterance(0, 0, 0, 250, 0),
            _utterance(1, 0, 300, 600, 1),
            _utterance(2, 1, 1000, 1300, 2),
        ],
        [[(0, 250), (300, 600)], [(1000, 1300)]],
        duration=1400,
    )
    relaxed_match = _analysis(
        [
            _utterance(0, 0, 0, 500, 0),
            _utterance(1, 1, 1800, 2100, 2),
        ],
        [[(0, 500)], [(1800, 2100)]],
        duration=2200,
    )
    metrics = _turn_event_metrics(source, relaxed_match, config)
    assert metrics["turn_event_tp"] == 2
    assert metrics["turn_event_fp"] == 0
    assert metrics["turn_event_fn"] == 0
    assert metrics["turn_event_f1"] == 1.0


def test_turn_event_matching_rejects_insufficient_source_overlap() -> None:
    config = InteractionConfig()
    source = _analysis(
        [
            _utterance(0, 0, 0, 200, 0),
            _utterance(1, 0, 250, 450, 1),
            _utterance(2, 0, 500, 700, 2),
        ],
        [[(0, 200), (250, 450), (500, 700)], []],
        duration=800,
    )
    reconstruction = _analysis(
        [_utterance(0, 0, 0, 700, 0)],
        [[(0, 700)], []],
        duration=800,
    )
    metrics = _turn_event_metrics(source, reconstruction, config)
    assert metrics["turn_event_tp"] == 0
    assert metrics["turn_event_fp"] == 1
    assert metrics["turn_event_fn"] == 1
    assert metrics["turn_event_f1"] == 0.0


def test_reconstruction_metrics_remove_duration_scale_from_overlap() -> None:
    source = _analysis(
        [_utterance(0, 0, 0, 600, 0), _utterance(1, 1, 500, 1000, 1)],
        [[(0, 600)], [(500, 1000)]],
        duration=1000,
    )
    reconstruction = _analysis(
        [_utterance(0, 0, 200, 1400, 0), _utterance(1, 1, 1200, 2200, 1)],
        [[(200, 1400)], [(1200, 2200)]],
        duration=2400,
    )
    result = _reconstruction_metrics(source, reconstruction, InteractionConfig())
    assert result["duration_ratio"] == 2.0
    assert result["overlap_event_f1"] == 1.0


def test_overlap_event_matching_uses_corresponding_utterance_pair() -> None:
    source = _analysis(
        [_utterance(0, 0, 0, 1000, 0), _utterance(1, 1, 400, 800, 1)],
        [[(0, 1000)], [(400, 800)]],
        duration=1000,
    )
    reconstruction = _analysis(
        [_utterance(0, 0, 0, 1000, 0), _utterance(1, 1, 700, 1000, 1)],
        [[(0, 1000)], [(700, 1000)]],
        duration=1000,
    )
    result = _reconstruction_metrics(source, reconstruction, InteractionConfig())
    assert result["overlap_event_f1"] == 1.0


def test_overlap_event_matching_merges_vad_fragments_for_same_utterance_pair() -> None:
    source = _analysis(
        [_utterance(0, 0, 0, 1000, 0), _utterance(1, 1, 400, 800, 1)],
        [[(0, 1000)], [(400, 800)]],
        duration=1000,
    )
    reconstruction = _analysis(
        [_utterance(0, 0, 0, 1000, 0), _utterance(1, 1, 400, 800, 1)],
        [[(0, 1000)], [(400, 550), (700, 800)]],
        duration=1000,
    )
    result = _reconstruction_metrics(source, reconstruction, InteractionConfig())
    assert result["overlap_event_tp"] == 1
    assert result["overlap_event_fp"] == 0
    assert result["overlap_event_fn"] == 0
    assert result["overlap_event_f1"] == 1.0


def test_backchannel_event_matching_uses_corresponding_utterance() -> None:
    source = _analysis(
        [
            InteractionUtterance(0, 0, 0, 1000, "main", source_index=0),
            InteractionUtterance(1, 1, 400, 550, "yeah", source_index=1),
        ],
        [[(0, 1000)], [(400, 550)]],
        duration=1000,
    )
    reconstruction = _analysis(
        [
            InteractionUtterance(0, 0, 0, 1000, "main", source_index=0),
            InteractionUtterance(1, 1, 1200, 1350, "yeah", source_index=1),
        ],
        [[(0, 1000)], [(1200, 1350)]],
        duration=1400,
    )
    result = _reconstruction_metrics(source, reconstruction, InteractionConfig())
    assert result["backchannel_event_f1"] == 1.0
