from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class PcmAudio:
    samples: np.ndarray
    size_bytes: int
    sha256: str


def read_speaker_wav(path: Path, *, duration_ms: int) -> PcmAudio:
    data = path.read_bytes()
    if not data:
        raise ValueError("speaker WAV is empty")
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != 16000
                or reader.getcomptype() != "NONE"
                or reader.getnframes() != duration_ms * 16
            ):
                raise ValueError("speaker WAV format or duration is invalid")
            frames = reader.readframes(reader.getnframes())
            if reader.readframes(1):
                raise ValueError("speaker WAV contains trailing frames")
    except (EOFError, wave.Error) as exc:
        raise ValueError("speaker WAV is malformed") from exc
    if len(frames) != duration_ms * 16 * 2:
        raise ValueError("speaker WAV PCM payload is truncated")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return PcmAudio(samples, len(data), hashlib.sha256(data).hexdigest())
