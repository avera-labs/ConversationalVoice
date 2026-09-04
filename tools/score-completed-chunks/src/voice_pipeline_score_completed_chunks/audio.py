from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

from .errors import ScoringError


@dataclass(frozen=True, slots=True)
class Audio:
    samples: np.ndarray
    sample_rate_hz: int
    frame_count: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ActiveAudio:
    samples: np.ndarray
    sample_rate_hz: int
    active_duration_ms: int
    interval_count: int


def read_wav(payload: bytes, *, expected_rate: int | None = None) -> Audio:
    try:
        with wave.open(io.BytesIO(payload), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            compression = reader.getcomptype()
            frames = reader.getnframes()
            raw = reader.readframes(frames)
    except (EOFError, wave.Error) as exc:
        raise ScoringError("invalid_wav") from exc
    if (
        channels != 1
        or width != 2
        or compression != "NONE"
        or frames <= 0
        or len(raw) != frames * 2
        or (expected_rate is not None and rate != expected_rate)
    ):
        raise ScoringError("invalid_wav_format")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return Audio(samples, rate, frames, round(frames * 1000 / rate))


def wav_bytes_from_pcm16(frames: bytes, *, sample_rate_hz: int) -> bytes:
    if not frames or len(frames) % 2:
        raise ScoringError("invalid_pcm16")
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(frames)
    return target.getvalue()


def wav_bytes_from_samples(samples: np.ndarray, *, sample_rate_hz: int) -> bytes:
    if samples.ndim != 1 or samples.size == 0 or sample_rate_hz <= 0:
        raise ScoringError("invalid_audio_samples")
    pcm = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")
    return wav_bytes_from_pcm16(pcm.tobytes(), sample_rate_hz=sample_rate_hz)


def mix_mono_tracks(left: Audio, right: Audio, *, target_rate: int = 16_000) -> bytes:
    """Mix equal-duration speaker tracks into a bounded mono WAV for ASR."""

    if left.duration_ms != right.duration_ms:
        raise ScoringError("track_duration_mismatch")
    left_samples = resample(left.samples, left.sample_rate_hz, target_rate)
    right_samples = resample(right.samples, right.sample_rate_hz, target_rate)
    frame_count = min(left_samples.size, right_samples.size)
    if frame_count == 0:
        raise ScoringError("interaction_empty_audio")
    mixed = left_samples[:frame_count] + right_samples[:frame_count]
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.999:
        mixed = mixed * (0.999 / peak)
    return wav_bytes_from_samples(mixed.astype(np.float32), sample_rate_hz=target_rate)


def slice_wav_payload(
    payload: bytes,
    *,
    segments: tuple[tuple[int, int], ...],
    expected_rate: int = 16000,
) -> bytes:
    source = read_wav(payload, expected_rate=expected_rate)
    pcm = np.clip(np.rint(source.samples * 32768.0), -32768, 32767).astype("<i2")
    selected: list[np.ndarray] = []
    for start_ms, end_ms in segments:
        if start_ms < 0 or end_ms <= start_ms:
            raise ScoringError("invalid_reference_selection")
        start = round(start_ms * expected_rate / 1000)
        end = round(end_ms * expected_rate / 1000)
        if end > source.frame_count or end <= start:
            raise ScoringError("reference_selection_out_of_bounds")
        selected.append(pcm[start:end])
    if not selected:
        raise ScoringError("empty_reference_selection")
    return wav_bytes_from_pcm16(
        np.concatenate(selected).astype("<i2", copy=False).tobytes(),
        sample_rate_hz=expected_rate,
    )


def slice_wav_interval(
    payload: bytes,
    *,
    start_ms: int,
    end_ms: int,
    expected_rate: int,
) -> bytes:
    """Return one timeline interval as a standalone mono PCM16 WAV."""

    source = read_wav(payload, expected_rate=expected_rate)
    if start_ms < 0 or end_ms <= start_ms or end_ms > source.duration_ms:
        raise ScoringError("utterance_interval_out_of_bounds")
    start = round(start_ms * source.sample_rate_hz / 1000)
    end = min(
        source.frame_count,
        round(end_ms * source.sample_rate_hz / 1000),
    )
    if end <= start:
        raise ScoringError("empty_utterance_audio")
    pcm = np.clip(np.rint(source.samples[start:end] * 32768.0), -32768, 32767).astype(
        "<i2"
    )
    return wav_bytes_from_pcm16(pcm.tobytes(), sample_rate_hz=source.sample_rate_hz)


def slice_audio(audio: Audio, *, start_ms: int, end_ms: int) -> Audio:
    if start_ms < 0 or end_ms <= start_ms or end_ms > audio.duration_ms:
        raise ScoringError("utterance_interval_out_of_bounds")
    start = round(start_ms * audio.sample_rate_hz / 1000)
    end = min(audio.frame_count, round(end_ms * audio.sample_rate_hz / 1000))
    if end <= start:
        raise ScoringError("empty_utterance_audio")
    samples = audio.samples[start:end].astype(np.float32, copy=False)
    return Audio(
        samples=samples,
        sample_rate_hz=audio.sample_rate_hz,
        frame_count=samples.size,
        duration_ms=round(samples.size * 1000 / audio.sample_rate_hz),
    )


def merge_intervals(
    intervals: list[tuple[int, int]], *, duration_ms: int
) -> tuple[tuple[int, int], ...]:
    if duration_ms <= 0:
        raise ScoringError("invalid_audio_duration")
    ordered = sorted(intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if start < 0 or end <= start or end > duration_ms:
            raise ScoringError("active_interval_out_of_bounds")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def extract_active_audio(
    audio: Audio,
    transcript: dict,
    *,
    speaker_id: int,
    separator_ms: int = 100,
) -> ActiveAudio:
    utterances = transcript.get("utterances")
    if not isinstance(utterances, list):
        raise ScoringError("invalid_transcript")
    intervals: list[tuple[int, int]] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            raise ScoringError("invalid_transcript")
        if utterance.get("speaker_id") != speaker_id:
            continue
        start = utterance.get("start_ms")
        end = utterance.get("end_ms")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ScoringError("invalid_transcript_interval")
        intervals.append((start, end))
    merged = merge_intervals(intervals, duration_ms=audio.duration_ms)
    if not merged:
        raise ScoringError("insufficient_active_speech")
    parts: list[np.ndarray] = []
    separator = np.zeros(round(separator_ms * audio.sample_rate_hz / 1000), np.float32)
    active_frames = 0
    for index, (start_ms, end_ms) in enumerate(merged):
        start = round(start_ms * audio.sample_rate_hz / 1000)
        end = min(audio.frame_count, round(end_ms * audio.sample_rate_hz / 1000))
        if end <= start:
            raise ScoringError("empty_active_interval")
        if index:
            parts.append(separator)
        parts.append(audio.samples[start:end])
        active_frames += end - start
    active_duration_ms = round(active_frames * 1000 / audio.sample_rate_hz)
    if active_duration_ms < 1000:
        raise ScoringError("insufficient_active_speech")
    return ActiveAudio(
        np.concatenate(parts).astype(np.float32, copy=False),
        audio.sample_rate_hz,
        active_duration_ms,
        len(merged),
    )


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    import torch
    import torchaudio

    tensor = torch.as_tensor(samples, dtype=torch.float32)
    result = torchaudio.functional.resample(tensor, source_rate, target_rate)
    return result.cpu().numpy().astype(np.float32, copy=False)
