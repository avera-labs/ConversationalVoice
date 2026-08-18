"""Exact normalized WAV slicing and speaker-reference assembly."""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .speaker_reference import ReferenceSegment

SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2


class WavError(ValueError):
    """Raised when input or generated audio violates the normalized contract."""


@dataclass(frozen=True, slots=True)
class PcmAudio:
    samples: np.ndarray
    sample_rate_hz: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class WrittenWav:
    sample_rate_hz: int
    duration_ms: int
    size_bytes: int
    sha256: str


def _duration_ms(frame_count: int, sample_rate_hz: int) -> int:
    numerator = frame_count * 1000
    duration_ms, remainder = divmod(numerator, sample_rate_hz)
    if remainder:
        raise WavError("WAV duration must have millisecond precision")
    return duration_ms


def _read_payload(path: Path) -> tuple[bytes, int]:
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != CHANNELS
                or reader.getsampwidth() != SAMPLE_WIDTH_BYTES
                or reader.getframerate() != SAMPLE_RATE_HZ
                or reader.getcomptype() != "NONE"
            ):
                raise WavError("WAV must be 16 kHz mono 16-bit PCM")
            frame_count = reader.getnframes()
            payload = reader.readframes(frame_count)
    except WavError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise WavError("WAV cannot be decoded") from exc
    if frame_count <= 0 or len(payload) != frame_count * SAMPLE_WIDTH_BYTES:
        raise WavError("WAV payload is empty or incomplete")
    return payload, frame_count


def read_normalized_wav(path: Path, *, expected_duration_ms: int) -> PcmAudio:
    payload, frame_count = _read_payload(path)
    duration_ms = _duration_ms(frame_count, SAMPLE_RATE_HZ)
    if duration_ms != expected_duration_ms:
        raise WavError("WAV duration does not match the audio part")
    return PcmAudio(
        samples=np.frombuffer(payload, dtype="<i2").copy(),
        sample_rate_hz=SAMPLE_RATE_HZ,
        duration_ms=duration_ms,
    )


def _milliseconds_to_frame(milliseconds: int) -> int:
    if milliseconds < 0:
        raise WavError("timestamp must not be negative")
    return milliseconds * (SAMPLE_RATE_HZ // 1000)


def write_reference_wav(
    audio: PcmAudio,
    destination: Path,
    segments: tuple[ReferenceSegment, ...],
    *,
    inter_segment_silence_ms: int,
) -> WrittenWav:
    if not segments or inter_segment_silence_ms < 0:
        raise WavError("reference WAV parameters are invalid")

    pieces: list[np.ndarray] = []
    silence_frames = _milliseconds_to_frame(inter_segment_silence_ms)
    for index, segment in enumerate(segments):
        start_frame = _milliseconds_to_frame(segment.start_ms)
        end_frame = _milliseconds_to_frame(segment.end_ms)
        if (
            start_frame < 0
            or end_frame <= start_frame
            or end_frame > audio.samples.size
        ):
            raise WavError("reference segment is outside the audio part")
        if index and silence_frames:
            pieces.append(np.zeros(silence_frames, dtype="<i2"))
        pieces.append(audio.samples[start_frame:end_frame])

    output = np.concatenate(pieces).astype("<i2", copy=False)
    expected_duration_ms = sum(segment.duration_ms for segment in segments)
    expected_duration_ms += inter_segment_silence_ms * (len(segments) - 1)
    if _duration_ms(output.size, SAMPLE_RATE_HZ) != expected_duration_ms:
        raise WavError("reference WAV duration is inconsistent")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(CHANNELS)
            writer.setsampwidth(SAMPLE_WIDTH_BYTES)
            writer.setframerate(SAMPLE_RATE_HZ)
            writer.writeframes(output.tobytes())
        payload, frame_count = _read_payload(destination)
        duration_ms = _duration_ms(frame_count, SAMPLE_RATE_HZ)
        if duration_ms != expected_duration_ms:
            raise WavError("written reference WAV duration is inconsistent")
        file_bytes = destination.read_bytes()
        if len(file_bytes) <= len(payload):
            raise WavError("written reference WAV header is invalid")
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return WrittenWav(
        sample_rate_hz=SAMPLE_RATE_HZ,
        duration_ms=duration_ms,
        size_bytes=len(file_bytes),
        sha256=hashlib.sha256(file_bytes).hexdigest(),
    )


__all__ = [
    "SAMPLE_RATE_HZ",
    "PcmAudio",
    "WavError",
    "WrittenWav",
    "read_normalized_wav",
    "write_reference_wav",
]
