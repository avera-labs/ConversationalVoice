from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

import numpy as np

from conftest import make_wav
from voice_pipeline_score_completed_chunks.repository import CompletedChunk
from voice_pipeline_score_completed_chunks.scoring import ScoreEngine
from voice_pipeline_score_completed_chunks.storage import StoredObject


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
        payload = self.objects[artifact.uri]
        assert len(payload) == artifact.size_bytes
        assert hashlib.sha256(payload).hexdigest() == artifact.sha256
        return payload


class FakeNisqa:
    def score(self, samples, sample_rate_hz):
        assert samples.size > sample_rate_hz
        return {
            "nisqa_mos": 4.0,
            "nisqa_noisiness": 4.1,
            "nisqa_discontinuity": 4.2,
            "nisqa_coloration": 4.3,
            "nisqa_loudness": 4.4,
        }


class FakeDnsmos:
    def score(self, samples, sample_rate_hz):
        return {
            "dnsmos_ovrl": 3.5,
            "dnsmos_sig": 3.6,
            "dnsmos_bak": 3.7,
            "dnsmos_p808": 3.8,
            "dnsmos_repeated_input": True,
            "dnsmos_window_count": 1,
        }


class FakeSpeaker:
    def embedding(self, audio):
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def similarity(self, output, reference):
        return float(np.dot(output, reference))


def test_score_engine_produces_six_speakers_and_three_groups() -> None:
    separation_tracks = [
        make_wav(duration_ms=4000, sample_rate_hz=16000, frequency_hz=200 + slot * 100)
        for slot in range(2)
    ]
    reconstruction_tracks = [
        make_wav(duration_ms=4000, sample_rate_hz=44100, frequency_hz=220 + slot * 100)
        for slot in range(2)
    ]
    expansion_tracks = [
        make_wav(duration_ms=4000, sample_rate_hz=44100, frequency_hz=240 + slot * 100)
        for slot in range(2)
    ]
    references = [
        make_wav(duration_ms=1500, sample_rate_hz=16000, frequency_hz=220 + slot * 100)
        for slot in range(2)
    ]
    mapping = [
        {"speaker_id": 0, "diarization_speaker_id": 10},
        {"speaker_id": 1, "diarization_speaker_id": 11},
    ]

    def transcript(timebase: str) -> bytes:
        value = {
            "language": "en",
            "timebase": timebase,
            "duration_ms": 4000,
            "speaker_mapping": mapping,
            "utterances": [
                {"speaker_id": 0, "start_ms": 0, "end_ms": 1500},
                {"speaker_id": 1, "start_ms": 1800, "end_ms": 3500},
            ],
        }
        return json.dumps(value).encode()

    separation_transcript = json.dumps(
        {
            "language": "en",
            "timebase": "chunk",
            "speakers": [
                {
                    "output_slot": 0,
                    "diarization_speaker_id": 10,
                    "utterances": [{"start_ms": 0, "end_ms": 1500}],
                },
                {
                    "output_slot": 1,
                    "diarization_speaker_id": 11,
                    "utterances": [{"start_ms": 1800, "end_ms": 3500}],
                },
            ],
        }
    ).encode()
    reconstruction_transcript = transcript("reconstruction")
    expansion_transcript = transcript("dialogue_extension")
    objects: dict[str, bytes] = {
        "s3://bucket/separation.json": separation_transcript,
        "s3://bucket/reconstruction.json": reconstruction_transcript,
        "s3://bucket/expansion.json": expansion_transcript,
    }
    for slot in range(2):
        objects[f"s3://bucket/separation-{slot}.wav"] = separation_tracks[slot]
        objects[f"s3://bucket/reconstruction-{slot}.wav"] = reconstruction_tracks[slot]
        objects[f"s3://bucket/expansion-{slot}.wav"] = expansion_tracks[slot]
        objects[f"s3://bucket/reference-{slot}.wav"] = references[slot]

    def tracks(group: str, payloads: list[bytes]) -> list[dict]:
        return [
            identity(
                f"s3://bucket/{group}-{slot}.wav",
                payload,
                speaker_id=slot,
                diarization_speaker_id=10 + slot,
                sample_rate_hz=44100,
                duration_ms=4000,
            )
            for slot, payload in enumerate(payloads)
        ]

    results = {
        "separation": {
            "speaker_audio": [
                identity(
                    f"s3://bucket/separation-{slot}.wav",
                    payload,
                    output_slot=slot,
                    diarization_speaker_id=10 + slot,
                    sample_rate_hz=16000,
                    duration_ms=4000,
                )
                for slot, payload in enumerate(separation_tracks)
            ]
        },
        "transcription": {
            "language": "en",
            "input_speaker_audio": [
                {
                    "output_slot": slot,
                    "diarization_speaker_id": 10 + slot,
                    **{
                        key: identity(
                            f"s3://bucket/separation-{slot}.wav",
                            separation_tracks[slot],
                        )[key]
                        for key in ("uri", "size_bytes", "sha256")
                    },
                }
                for slot in range(2)
            ],
            "artifacts": {
                "transcript": identity(
                    "s3://bucket/separation.json", separation_transcript
                )
            },
        },
        "reconstruction": {
            "language": "en",
            "actual_duration_ms": 4000,
            "artifacts": {
                "transcript": identity(
                    "s3://bucket/reconstruction.json", reconstruction_transcript
                ),
                "speaker_audio": tracks("reconstruction", reconstruction_tracks),
            },
        },
        "dialogue_extension": {
            "language": "en",
            "actual_duration_ms": 4000,
            "artifacts": {
                "transcript": identity(
                    "s3://bucket/expansion.json", expansion_transcript
                ),
                "speaker_audio": tracks("expansion", expansion_tracks),
            },
            "inputs": {
                "speaker_references": [
                    {
                        "speaker_id": slot,
                        "diarization_speaker_id": 10 + slot,
                        "source": "diarization_reference",
                        "source_audio": identity(
                            f"s3://bucket/reference-{slot}.wav", reference
                        ),
                        "selection": {
                            "timebase": "audio_part",
                            "segments": [
                                {"start_ms": 0, "end_ms": 1500, "duration_ms": 1500}
                            ],
                        },
                        "reference_audio": {
                            "sample_rate_hz": 16000,
                            "duration_ms": 1500,
                            "size_bytes": len(reference),
                            "sha256": hashlib.sha256(reference).hexdigest(),
                        },
                    }
                    for slot, reference in enumerate(references)
                ]
            },
        },
    }
    chunk = CompletedChunk(
        UUID("00000000-0000-0000-0000-000000000001"),
        "en",
        datetime.now(UTC),
        "s3://bucket/audio-part.wav",
        results,
    )
    engine = ScoreEngine(
        FakeStorage(objects),  # type: ignore[arg-type]
        FakeNisqa(),  # type: ignore[arg-type]
        FakeDnsmos(),  # type: ignore[arg-type]
        FakeSpeaker(),  # type: ignore[arg-type]
        "fingerprint",
    )
    speaker_rows, group_rows, failures = engine.score_chunk(chunk, existing={})
    assert len(speaker_rows) == 6
    assert len(group_rows) == 3
    assert not failures
    assert all(row["status"] == "success" for row in speaker_rows)
    assert all(row["valid_speaker_count"] == 2 for row in group_rows)
