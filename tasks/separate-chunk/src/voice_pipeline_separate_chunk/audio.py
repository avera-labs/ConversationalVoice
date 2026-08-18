from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True, slots=True)
class Audio:
    samples: np.ndarray
    sample_rate: int
    size_bytes: int
    sha256: str


def read_chunk(path: Path, duration_ms: int) -> Audio:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 16000
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError("invalid chunk WAV")
        frames = reader.getnframes()
        raw = reader.readframes(frames)
    if frames != round(duration_ms * 16):
        raise ValueError("chunk duration mismatch")
    payload = path.read_bytes()
    return Audio(
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768,
        16000,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def write_outputs(
    tracks: np.ndarray,
    native_rate: int,
    paths: tuple[Path, Path],
    duration_ms: int,
    peak: float,
) -> tuple[dict[str, object], ...]:
    import torch
    import torchaudio

    tensor = torch.as_tensor(tracks, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] != 2 or not torch.isfinite(tensor).all():
        raise ValueError("invalid output tracks")
    if native_rate != 16000:
        tensor = torchaudio.functional.resample(tensor, native_rate, 16000)
    target = round(duration_ms * 16)
    if tensor.shape[1] < target:
        tensor = torch.nn.functional.pad(tensor, (0, target - tensor.shape[1]))
    tensor = tensor[:, :target]
    if not torch.isfinite(tensor).all():
        raise ValueError("invalid resampled output tracks")
    maximum = float(tensor.abs().max())
    if maximum > 0:
        tensor = tensor * (peak / maximum)
    result = []
    for slot, path in enumerate(paths):
        sf.write(path, tensor[slot].numpy(), 16000, subtype="PCM_16")
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != 16000
                or reader.getnframes() != target
                or reader.getcomptype() != "NONE"
            ):
                raise ValueError("invalid output WAV")
        payload = path.read_bytes()
        result.append(
            {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    return tuple(result)
