from __future__ import annotations

import os
import tracemalloc
import wave
from pathlib import Path

import pytest

from voice_pipeline_split_raw_audio_into_parts.wav_io import (
    CHANNEL_COUNT,
    COPY_BLOCK_FRAMES,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    cut_wav_frames,
)


pytestmark = pytest.mark.capacity

if os.environ.get("RUN_CAPACITY_TESTS") != "1":
    pytest.skip(
        "Set RUN_CAPACITY_TESTS=1 through tests/capacity/run.sh.",
        allow_module_level=True,
    )


def _write_silent_wav(path: Path, frame_count: int) -> None:
    block = bytes(COPY_BLOCK_FRAMES * SAMPLE_WIDTH_BYTES)
    remaining = frame_count
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(CHANNEL_COUNT)
        writer.setsampwidth(SAMPLE_WIDTH_BYTES)
        writer.setframerate(SAMPLE_RATE)
        while remaining:
            frames = min(remaining, COPY_BLOCK_FRAMES)
            writer.writeframesraw(block[: frames * SAMPLE_WIDTH_BYTES])
            remaining -= frames


def test_nine_hundred_second_clip_is_streamed_with_bounded_memory(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.wav"
    destination_path = tmp_path / "part.wav"
    source_frame_count = SAMPLE_RATE * 901
    clip_frame_count = SAMPLE_RATE * 900
    _write_silent_wav(source_path, source_frame_count)

    tracemalloc.start()
    try:
        clip = cut_wav_frames(
            source_path,
            destination_path,
            start_frame=0,
            end_frame=clip_frame_count,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    with wave.open(str(destination_path), "rb") as reader:
        assert reader.getnframes() == clip_frame_count
        assert reader.getframerate() == SAMPLE_RATE
        assert reader.getnchannels() == CHANNEL_COUNT
        assert reader.getsampwidth() == SAMPLE_WIDTH_BYTES

    assert clip.duration_ms == 900_000
    assert peak_bytes < 2 * 1024 * 1024
