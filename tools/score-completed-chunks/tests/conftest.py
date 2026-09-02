from __future__ import annotations

import io
import wave

import numpy as np


def make_wav(
    *,
    duration_ms: int = 2000,
    sample_rate_hz: int = 16000,
    frequency_hz: float = 220.0,
) -> bytes:
    count = round(duration_ms * sample_rate_hz / 1000)
    time = np.arange(count, dtype=np.float64) / sample_rate_hz
    samples = np.rint(np.sin(2 * np.pi * frequency_hz * time) * 8000).astype("<i2")
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(samples.tobytes())
    return target.getvalue()
