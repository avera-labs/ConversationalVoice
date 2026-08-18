from __future__ import annotations

import hashlib
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioIdentity:
    size_bytes: int
    sha256: str


def validate_wav(path: Path, *, duration_ms: int) -> AudioIdentity:
    data = path.read_bytes()
    if not data:
        raise ValueError("chunk WAV is empty")
    try:
        with wave.open(str(path), "rb") as reader:
            valid = (
                reader.getnchannels() == 1
                and reader.getsampwidth() == 2
                and reader.getframerate() == 16000
                and reader.getcomptype() == "NONE"
                and reader.getnframes() == duration_ms * 16
            )
    except (EOFError, wave.Error) as exc:
        raise ValueError("chunk WAV is invalid") from exc
    if not valid:
        raise ValueError("chunk WAV format or duration is invalid")
    return AudioIdentity(len(data), hashlib.sha256(data).hexdigest())


def encode_mp3(source: Path, destination: Path, policy) -> int:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(policy.channels),
        "-ar",
        str(policy.sample_rate_hz),
        "-b:a",
        f"{policy.bitrate_kbps}k",
        "-f",
        policy.format,
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError("audio_encoding_failed") from exc
    size = destination.stat().st_size if destination.exists() else 0
    if size <= 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError("audio_encoding_produced_empty_output")
    return size
