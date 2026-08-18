from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path

import pytest
from dotenv import load_dotenv

from voice_pipeline_split_raw_audio_into_parts.config import (
    EnvironmentSettings,
    VadPolicy,
)
from voice_pipeline_split_raw_audio_into_parts.vad import (
    PyannoteVad,
    clear_vad_pipeline_cache,
)
from voice_pipeline_split_raw_audio_into_parts.wav_io import (
    CHANNEL_COUNT,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
)


pytestmark = pytest.mark.model_smoke

if os.environ.get("RUN_MODEL_SMOKE_TEST") != "1":
    pytest.skip(
        "Set RUN_MODEL_SMOKE_TEST=1 through tests/smoke/run.sh.",
        allow_module_level=True,
    )


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing model smoke environment variable: {name}.")
    return value


def test_real_pyannote_segmentation_inference(tmp_path: Path) -> None:
    load_dotenv(Path(__file__).parents[4] / ".env.test", override=True)
    environment = EnvironmentSettings(
        database_url=_required_environment("TEST_DATABASE_URL"),
        celery_broker_url=_required_environment("TEST_CELERY_BROKER_URL"),
        s3_bucket=_required_environment("TEST_S3_BUCKET"),
        s3_region=_required_environment("TEST_S3_REGION"),
        s3_endpoint_url=_required_environment("TEST_S3_ENDPOINT_URL"),
        hf_token=_required_environment("TEST_HF_TOKEN"),
    )
    device = os.environ.get("TEST_VAD_DEVICE", "cpu")
    policy = VadPolicy(
        model="pyannote/segmentation-3.0",
        device=device,
    )

    frame_count = SAMPLE_RATE * 2
    samples = [
        int(2_000 * math.sin(2 * math.pi * 220 * index / SAMPLE_RATE))
        for index in range(frame_count)
    ]
    audio_path = tmp_path / "model-smoke.wav"
    with wave.open(str(audio_path), "wb") as writer:
        writer.setnchannels(CHANNEL_COUNT)
        writer.setsampwidth(SAMPLE_WIDTH_BYTES)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(struct.pack(f"<{frame_count}h", *samples))

    try:
        vad = PyannoteVad.create(policy, environment)
        result = vad.run(audio_path)
    finally:
        clear_vad_pipeline_cache()

    assert result.model == "pyannote/segmentation-3.0"
    assert result.audio_frame_count == frame_count
    assert all(
        0 <= segment.start_frame < segment.end_frame <= frame_count
        for segment in result.segments
    )
