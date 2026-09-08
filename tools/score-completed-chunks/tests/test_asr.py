from __future__ import annotations

import base64
import os

import numpy as np
import pytest

from voice_pipeline_score_completed_chunks.asr import (
    OpenRouterAsrClient,
    TranscriptionResult,
    error_rate,
)
from voice_pipeline_score_completed_chunks.audio import wav_bytes_from_samples
from voice_pipeline_score_completed_chunks.service import ChunkScoreService


class FakeResponse:
    status_code = 200
    headers = {"X-Generation-Id": "generation-1"}

    @staticmethod
    def json():
        return {"text": "hello world", "usage": {"seconds": 1.0}}


class FakeSession:
    def __init__(self):
        self.request = None

    def post(self, endpoint, **kwargs):
        self.request = (endpoint, kwargs)
        return FakeResponse()

    def close(self):
        return None


def test_error_rate_uses_cer_for_chinese_and_wer_for_english() -> None:
    chinese = error_rate("你好，世界", "你好世", language="zh-CN")
    english = error_rate("Hello, brave world!", "hello world", language="en")
    assert chinese["metric"] == "cer"
    assert chinese["value"] == 0.25
    assert chinese["edit_count"] == 1
    assert chinese["deletions"] == 1
    assert chinese["normalized_reference"] == "你好世界"
    assert chinese["normalized_hypothesis"] == "你好世"
    assert english["metric"] == "wer"
    assert english["value"] == pytest.approx(1 / 3)


def test_openrouter_asr_uses_dedicated_json_endpoint() -> None:
    session = FakeSession()
    client = OpenRouterAsrClient("secret", session=session)  # type: ignore[arg-type]
    result = client.transcribe(b"RIFFpayload", language="zh-CN")
    assert result.text == "hello world"
    assert result.generation_id == "generation-1"
    _endpoint, kwargs = session.request
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["json"]["model"] == "qwen/qwen3-asr-1.7b"
    assert kwargs["json"]["language"] == "zh"
    assert kwargs["json"]["provider"] == {"data_collection": "deny", "zdr": True}
    assert base64.b64decode(kwargs["json"]["input_audio"]["data"]) == b"RIFFpayload"


def test_long_audio_is_split_at_utterance_boundaries() -> None:
    class Asr:
        model = "test/asr"

        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, *, language):
            self.calls += 1
            return TranscriptionResult(
                f"segment {self.calls}",
                self.model,
                f"generation-{self.calls}",
                {"seconds": 40.0},
            )

    service = object.__new__(ChunkScoreService)
    service.asr = Asr()
    audio = wav_bytes_from_samples(np.zeros(120 * 16_000), sample_rate_hz=16_000)
    result = service._transcribe_windows(
        audio,
        utterances=[{"end_ms": value} for value in (40_000, 80_000, 120_000)],
        language="en",
    )
    assert service.asr.calls == 3
    assert result["hypothesis"] == "segment 1 segment 2 segment 3"
    assert [item["end_ms"] for item in result["segments"]] == [
        40_000,
        80_000,
        120_000,
    ]


@pytest.mark.provider_smoke
@pytest.mark.skipif(
    os.environ.get("RUN_OPENROUTER_ASR_SMOKE") != "1",
    reason="paid provider smoke test is disabled",
)
def test_openrouter_asr_provider_smoke() -> None:
    key = os.environ["OPENROUTER_API_KEY"]
    samples = np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000) * 0.05
    audio = wav_bytes_from_samples(samples.astype(np.float32), sample_rate_hz=16_000)
    result = OpenRouterAsrClient(key).transcribe(audio, language="en")
    assert isinstance(result.text, str)
    assert result.model == "qwen/qwen3-asr-1.7b"
