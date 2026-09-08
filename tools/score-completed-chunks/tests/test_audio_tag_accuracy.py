from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from conftest import make_wav
from voice_pipeline_score_completed_chunks.audio_tag_accuracy import (
    AudioTagEvaluator,
    AudioTagScoreEngine,
    build_annotation_record,
    summarize_audio_tag_rows,
)
from voice_pipeline_score_completed_chunks.repository import CompletedChunk
from voice_pipeline_score_completed_chunks.storage import StoredObject


def identity(uri: str, payload: bytes, **extra) -> dict:
    return {
        "uri": uri,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        **extra,
    }


class FakeResponse:
    status_code = 200

    def json(self):
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"score": 4, "reason": "The tag is clearly audible."}
                        )
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "cost": 0.001,
            },
        }


class FakeTransport:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


def test_openrouter_evaluator_sends_tagged_text_and_wav() -> None:
    transport = FakeTransport()
    evaluator = AudioTagEvaluator("secret", transport=transport)
    result = evaluator.evaluate(
        audio=make_wav(duration_ms=500, sample_rate_hz=44100),
        annotation_record={
            "representation": "inline_text_with_audio_tags",
            "text_with_audio_tags": "[laughs]Hello.",
        },
        tags=("[laughs]",),
        language="en",
    )

    assert result["score"] == 4
    assert result["usage"]["total_tokens"] == 20
    _, request = transport.calls[0]
    payload = request["json"]
    assert payload["model"] == "google/gemini-3.7-flash"
    assert payload["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
    assert payload["seed"] == 0
    assert payload["reasoning"] == {"effort": "low", "exclude": True}
    assert payload["max_tokens"] == 2048
    assert "temperature" not in payload
    assert "response_format" not in payload
    user_content = payload["messages"][1]["content"]
    assert "[laughs]Hello." in user_content[0]["text"]
    assert user_content[1]["type"] == "input_audio"
    assert user_content[1]["input_audio"]["format"] == "wav"


def test_legacy_tone_and_audio_tags_are_preserved_without_fake_positions() -> None:
    record, tags, representation = build_annotation_record(
        {
            "text": "Hello.",
            "tone": "warm",
            "audio_tags": ["laughing"],
        }
    )

    assert representation == "legacy_text_tone_audio_tags"
    assert record == {
        "representation": "legacy_text_tone_audio_tags",
        "text": "Hello.",
        "tone": "warm",
        "audio_tags": ["laughing"],
    }
    assert tags == ("tone:warm", "audio_tag:laughing")


class FakeStorage:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def download(self, artifact: StoredObject) -> bytes:
        return self.objects[artifact.uri]


class FakeEvaluator:
    model = "google/gemini-3.7-flash"
    fingerprint = "evaluator-fingerprint"

    def __init__(self):
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return {"score": 5, "reason": "Perfect match.", "usage": {}}


def test_score_engine_evaluates_tagged_utterances_and_excludes_untagged() -> None:
    duration_ms = 2000
    tracks = {
        (group, speaker): make_wav(
            duration_ms=duration_ms,
            sample_rate_hz=44100,
            frequency_hz=220 + speaker * 100,
        )
        for group in ("reconstruction", "expansion")
        for speaker in range(2)
    }
    mapping = [
        {"speaker_id": 0, "diarization_speaker_id": 10},
        {"speaker_id": 1, "diarization_speaker_id": 11},
    ]
    objects: dict[str, bytes] = {}
    final_results = {}
    for group, namespace, timebase in (
        ("reconstruction", "reconstruction", "reconstruction"),
        ("expansion", "dialogue_extension", "dialogue_extension"),
    ):
        transcript = json.dumps(
            {
                "language": "en",
                "timebase": timebase,
                "duration_ms": duration_ms,
                "speaker_mapping": mapping,
                "utterances": [
                    {
                        "utterance_index": 7,
                        "speaker_id": 0,
                        "start_ms": 0,
                        "end_ms": 800,
                        "text_with_audio_tags": "[calm]Hello.",
                    },
                    {
                        "utterance_index": 8,
                        "speaker_id": 1,
                        "start_ms": 1000,
                        "end_ms": 1800,
                        "text": "Goodbye.",
                        "tone": "neutral",
                        "audio_tags": [],
                    },
                ],
            }
        ).encode()
        transcript_uri = f"s3://bucket/{group}.json"
        objects[transcript_uri] = transcript
        speaker_audio = []
        for speaker in range(2):
            uri = f"s3://bucket/{group}-{speaker}.wav"
            objects[uri] = tracks[(group, speaker)]
            speaker_audio.append(
                identity(
                    uri,
                    tracks[(group, speaker)],
                    speaker_id=speaker,
                    diarization_speaker_id=10 + speaker,
                    sample_rate_hz=44100,
                    duration_ms=duration_ms,
                )
            )
        final_results[namespace] = {
            "language": "en",
            "actual_duration_ms": duration_ms,
            "artifacts": {
                "transcript": identity(transcript_uri, transcript),
                "speaker_audio": speaker_audio,
            },
        }

    chunk = CompletedChunk(
        UUID("00000000-0000-0000-0000-000000000001"),
        "en",
        datetime.now(UTC),
        "s3://bucket/source.wav",
        final_results,
    )
    evaluator = FakeEvaluator()
    engine = AudioTagScoreEngine(
        FakeStorage(objects),  # type: ignore[arg-type]
        evaluator,  # type: ignore[arg-type]
        workers=2,
        inline_only=True,
    )
    rows, failures = engine.score_chunk(chunk, existing={})

    assert not failures
    assert len(rows) == 4
    assert len(evaluator.calls) == 2
    assert {row["status"] for row in rows} == {"success", "not_applicable"}
    assert all(
        row["reason"] == "inline text_with_audio_tags is unavailable"
        for row in rows
        if row["status"] == "not_applicable"
    )
    assert all(call["audio"].startswith(b"RIFF") for call in evaluator.calls)
    summary = summarize_audio_tag_rows(rows)
    assert summary["overall"]["evaluated_count"] == 2
    assert summary["overall"]["not_applicable_count"] == 2
    assert summary["overall"]["score_counts"]["5"] == 2
    assert summary["overall"]["perfect_match_rate"] == 1.0
