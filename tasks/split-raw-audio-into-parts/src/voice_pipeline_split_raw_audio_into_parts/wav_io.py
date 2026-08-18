from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000
CHANNEL_COUNT = 1
SAMPLE_WIDTH_BYTES = 2
COPY_BLOCK_FRAMES = 65_536


class WavCutError(ValueError):
    """Raised when a requested PCM frame range cannot be cut completely."""


@dataclass(frozen=True, slots=True)
class WavClip:
    path: Path
    start_frame: int
    end_frame: int
    frame_count: int
    relative_start_ms: int
    relative_end_ms: int
    duration_ms: int


def frame_to_milliseconds(frame: int) -> int:
    if frame < 0:
        raise ValueError("frame must not be negative")
    return (frame * 1_000 + SAMPLE_RATE // 2) // SAMPLE_RATE


def milliseconds_to_frames(milliseconds: int) -> int:
    if milliseconds < 0:
        raise ValueError("milliseconds must not be negative")
    return (milliseconds * SAMPLE_RATE + 500) // 1_000


def cut_wav_frames(
    source_path: Path,
    destination_path: Path,
    *,
    start_frame: int,
    end_frame: int,
) -> WavClip:
    """Copy one non-empty frame range without decoding or resampling the WAV."""

    if start_frame < 0:
        raise WavCutError("start_frame must not be negative")
    if end_frame <= start_frame:
        raise WavCutError("end_frame must be greater than start_frame")
    if source_path.resolve() == destination_path.resolve():
        raise WavCutError("source and destination paths must differ")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(source_path), "rb") as reader:
            source_frame_count = reader.getnframes()
            if end_frame > source_frame_count:
                raise WavCutError("requested frame range exceeds the source WAV")

            parameters = reader.getparams()
            bytes_per_frame = parameters.nchannels * parameters.sampwidth
            reader.setpos(start_frame)
            remaining = end_frame - start_frame
            written_frames = 0

            with wave.open(str(destination_path), "wb") as writer:
                writer.setparams(parameters)
                while remaining:
                    requested = min(remaining, COPY_BLOCK_FRAMES)
                    data = reader.readframes(requested)
                    if not data or len(data) % bytes_per_frame:
                        raise WavCutError("source WAV ended inside the requested range")
                    copied = len(data) // bytes_per_frame
                    writer.writeframesraw(data)
                    written_frames += copied
                    remaining -= copied

            if written_frames != end_frame - start_frame:
                raise WavCutError("output WAV frame count is incomplete")
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise

    relative_start_ms = frame_to_milliseconds(start_frame)
    relative_end_ms = frame_to_milliseconds(end_frame)
    return WavClip(
        path=destination_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_count=end_frame - start_frame,
        relative_start_ms=relative_start_ms,
        relative_end_ms=relative_end_ms,
        duration_ms=relative_end_ms - relative_start_ms,
    )
