from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from conftest import make_wav
from voice_pipeline_score_completed_chunks.conversation_quality import (
    ConversationQualityEvaluationError,
    ConversationQualityEvaluator,
    ConversationQualityScoreEngine,
    _parse_response,
    summarize_conversation_quality_rows,
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
                            {
                                "content_coherence": {
                                    "score": 4.8,
                                    "reason": "The continuation stays on topic.",
                                },
                                "dialogue_naturalness": {
                                    "score": 4.7,
                                    "reason": "The exchange is spontaneous.",
                                },
                            }
                        )
                    },
                }
            ],
            "usage": {"total_tokens": 30, "cost": 0.002},
        }


class FakeTransport:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


class FencedFakeResponse(FakeResponse):
    def json(self):
        value = super().json()
        raw = value["choices"][0]["message"]["content"]
        value["choices"][0]["message"]["content"] = f"```json\n{raw}\n```"
        return value


class FencedFakeTransport(FakeTransport):
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FencedFakeResponse()


def test_evaluator_sends_two_wavs_in_one_request_and_returns_two_scores() -> None:
    transport = FakeTransport()
    evaluator = ConversationQualityEvaluator("secret", transport=transport)
    result = evaluator.evaluate(
        reconstruction_audio=make_wav(duration_ms=500),
        expansion_audio=make_wav(duration_ms=800),
        language="en",
    )

    assert result["content_coherence"]["score"] == 4.8
    assert result["dialogue_naturalness"]["score"] == 4.7
    assert result["usage"]["cost_usd"] == 0.002
    assert len(transport.calls) == 1
    payload = transport.calls[0][1]["json"]
    assert payload["model"] == "google/gemini-3.7-flash"
    assert payload["provider"]["data_collection"] == "deny"
    assert payload["provider"]["zdr"] is True
    assert payload["seed"] == 0
    content = payload["messages"][1]["content"]
    audio_parts = [part for part in content if part["type"] == "input_audio"]
    assert len(audio_parts) == 2
    assert all(part["input_audio"]["format"] == "wav" for part in audio_parts)
    assert "prefer a fine-grained non-integer score, such as 4.8" in payload[
        "messages"
    ][0]["content"]
    assert "give exactly one concise" in payload["messages"][0]["content"]


def test_evaluator_accepts_json_code_fence_but_still_validates_fields() -> None:
    evaluator = ConversationQualityEvaluator(
        "secret", transport=FencedFakeTransport()
    )

    result = evaluator.evaluate(
        reconstruction_audio=make_wav(duration_ms=500),
        expansion_audio=make_wav(duration_ms=800),
        language="en",
    )

    assert result["content_coherence"]["score"] == 4.8
    assert result["dialogue_naturalness"]["score"] == 4.7


@pytest.mark.parametrize("invalid_score", [5, 4.85, 0.9, 5.1])
def test_response_rejects_non_float_or_non_tenth_scores(invalid_score) -> None:
    response = FakeResponse().json()
    content = json.loads(response["choices"][0]["message"]["content"])
    content["content_coherence"]["score"] = invalid_score
    response["choices"][0]["message"]["content"] = json.dumps(content)

    with pytest.raises(ConversationQualityEvaluationError):
        _parse_response(response)


class FakeStorage:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def download(self, artifact: StoredObject) -> bytes:
        return self.objects[artifact.uri]


class FakeEvaluator:
    model = "google/gemini-test"
    fingerprint = "fingerprint"

    def __init__(self):
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "content_coherence": {"score": 4.9, "reason": "Coherent."},
            "dialogue_naturalness": {"score": 4.6, "reason": "Natural."},
            "usage": {},
        }


def test_engine_mixes_both_stages_and_calls_evaluator_once() -> None:
    duration_ms = 1000
    objects: dict[str, bytes] = {}
    final_results = {}
    for group, namespace in (
        ("reconstruction", "reconstruction"),
        ("expansion", "dialogue_extension"),
    ):
        tracks = []
        for speaker in range(2):
            payload = make_wav(
                duration_ms=duration_ms,
                sample_rate_hz=44100,
                frequency_hz=220 + speaker * 100,
            )
            uri = f"s3://bucket/{group}-{speaker}.wav"
            objects[uri] = payload
            tracks.append(
                identity(
                    uri,
                    payload,
                    speaker_id=speaker,
                    diarization_speaker_id=10 + speaker,
                    sample_rate_hz=44100,
                    duration_ms=duration_ms,
                )
            )
        transcript = b"{}"
        transcript_uri = f"s3://bucket/{group}.json"
        objects[transcript_uri] = transcript
        final_results[namespace] = {
            "language": "en",
            "actual_duration_ms": duration_ms,
            "artifacts": {
                "transcript": identity(transcript_uri, transcript),
                "speaker_audio": tracks,
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
    engine = ConversationQualityScoreEngine(
        FakeStorage(objects),  # type: ignore[arg-type]
        evaluator,  # type: ignore[arg-type]
    )

    row, reused = engine.score_chunk(chunk)

    assert reused is False
    assert row["content_coherence"]["score"] == 4.9
    assert row["dialogue_naturalness"]["score"] == 4.6
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0]["language"] == "en"
    assert row["audio"]["reconstruction"]["submitted_sample_rate_hz"] == 16000
    reused_row, reused = engine.score_chunk(chunk, existing=row)
    assert reused is True
    assert reused_row == row
    assert len(evaluator.calls) == 1


def test_summary_averages_successful_chunk_scores() -> None:
    rows = [
        {
            "status": "success",
            "content_coherence": {"score": 4.8},
            "dialogue_naturalness": {"score": 4.6},
            "usage": {"cost_usd": 0.01},
        },
        {
            "status": "success",
            "content_coherence": {"score": 4.2},
            "dialogue_naturalness": {"score": 3.8},
            "usage": {"cost_usd": 0.02},
        },
    ]

    summary = summarize_conversation_quality_rows(rows, requested_count=3)

    assert summary["successful_chunk_count"] == 2
    assert summary["failed_chunk_count"] == 1
    assert summary["metrics"]["content_coherence"]["mean"] == 4.5
    assert summary["metrics"]["dialogue_naturalness"]["mean"] == 4.2
    assert summary["metrics"]["content_coherence"]["score_counts"] == {
        "4.2": 1,
        "4.8": 1,
    }
    assert summary["total_cost_usd"] == 0.03
