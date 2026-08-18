"""Strict 16 kHz mono PCM WAV decoding and millisecond-exact slicing."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .intervals import rational_to_milliseconds

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2


class WavError(ValueError):
    """Raised when a WAV violates the normalized audio contract."""


@dataclass(frozen=True, slots=True)
class PcmAudio:
    samples: np.ndarray
    sample_rate: int
    duration_ms: int

    @property
    def waveform(self) -> np.ndarray:
        return self.samples.astype(np.float32) / 32768.0


def milliseconds_to_frame(milliseconds: int) -> int:
    if milliseconds < 0:
        raise ValueError("milliseconds must not be negative")
    return milliseconds * (SAMPLE_RATE // 1000)


def read_normalized_wav(path: Path, *, expected_duration_ms: int) -> PcmAudio:
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != CHANNELS
                or reader.getsampwidth() != SAMPLE_WIDTH_BYTES
                or reader.getframerate() != SAMPLE_RATE
                or reader.getcomptype() != "NONE"
            ):
                raise WavError("WAV format is not 16 kHz mono 16-bit PCM")
            frame_count = reader.getnframes()
            payload = reader.readframes(frame_count)
            if len(payload) != frame_count * SAMPLE_WIDTH_BYTES:
                raise WavError("WAV payload is incomplete")
    except (OSError, EOFError, wave.Error) as exc:
        raise WavError("WAV cannot be decoded") from exc
    duration_ms = rational_to_milliseconds(frame_count, SAMPLE_RATE)
    if duration_ms != expected_duration_ms:
        raise WavError("WAV duration does not match the database")
    samples = np.frombuffer(payload, dtype="<i2").copy()
    if samples.size == 0:
        raise WavError("WAV is empty")
    return PcmAudio(samples=samples, sample_rate=SAMPLE_RATE, duration_ms=duration_ms)


def speech_samples(audio: PcmAudio, *, start_ms: int, end_ms: int) -> np.ndarray:
    start_frame = milliseconds_to_frame(start_ms)
    end_frame = milliseconds_to_frame(end_ms)
    if start_frame < 0 or end_frame <= start_frame or end_frame > audio.samples.size:
        raise WavError("speech interval is outside the WAV")
    return audio.waveform[start_frame:end_frame]


def write_chunk_wav(
    audio: PcmAudio, destination: Path, *, start_ms: int, end_ms: int
) -> int:
    start_frame = milliseconds_to_frame(start_ms)
    end_frame = milliseconds_to_frame(end_ms)
    if start_frame < 0 or end_frame <= start_frame or end_frame > audio.samples.size:
        raise WavError("chunk interval is outside the WAV")
    selected = audio.samples[start_frame:end_frame]
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(CHANNELS)
            writer.setsampwidth(SAMPLE_WIDTH_BYTES)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(selected.astype("<i2", copy=False).tobytes())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    duration_ms = rational_to_milliseconds(selected.size, SAMPLE_RATE)
    if duration_ms != end_ms - start_ms:
        destination.unlink(missing_ok=True)
        raise WavError("chunk WAV duration is inconsistent")
    return destination.stat().st_size
