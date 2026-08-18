import math
import os
import struct
import subprocess
import wave

import pytest

from voice_pipeline_persona_chunk.config import load_settings
from voice_pipeline_persona_chunk.openrouter import OpenRouterClient

pytestmark = pytest.mark.provider_smoke
if os.environ.get("RUN_OPENROUTER_SMOKE") != "1":
    pytest.skip(
        "Set RUN_OPENROUTER_SMOKE=1 through tests/smoke/run.sh.",
        allow_module_level=True,
    )


def test_configured_model_accepts_audio_transcript_and_structured_output(tmp_path):
    settings = load_settings()
    wav = tmp_path / "fixture.wav"
    mp3 = tmp_path / "fixture.mp3"
    with wave.open(str(wav), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        frames = b"".join(
            struct.pack("<h", round(1200 * math.sin(2 * math.pi * 220 * i / 16000)))
            for i in range(32000)
        )
        writer.writeframes(frames)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "48k",
            str(mp3),
        ],
        check=True,
    )
    srt = (
        "1\n00:00:00,000 --> 00:00:01,000\n[Speaker 4]: Hello.\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n[Speaker 7]: Hi.\n"
    )
    client = OpenRouterClient(
        settings.policy.openrouter,
        settings.environment.openrouter_api_key.get_secret_value(),
    )
    try:
        _wire, usage = client.analyze(mp3.read_bytes(), srt, (4, 7))
        assert usage["model"] == settings.policy.openrouter.model
    finally:
        client.close()
