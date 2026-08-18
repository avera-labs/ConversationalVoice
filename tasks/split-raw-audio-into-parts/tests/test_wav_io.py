import struct
import wave
from pathlib import Path

import pytest

from voice_pipeline_split_raw_audio_into_parts.wav_io import (
    CHANNEL_COUNT,
    COPY_BLOCK_FRAMES,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    WavCutError,
    cut_wav_frames,
    frame_to_milliseconds,
    milliseconds_to_frames,
)


def _write_wav(path: Path, samples: list[int]) -> None:
    payload = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(CHANNEL_COUNT)
        writer.setsampwidth(SAMPLE_WIDTH_BYTES)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(payload)


def _read_samples(path: Path) -> tuple[wave._wave_params, list[int]]:
    with wave.open(str(path), "rb") as reader:
        parameters = reader.getparams()
        payload = reader.readframes(reader.getnframes())
    samples = list(struct.unpack(f"<{len(payload) // 2}h", payload))
    return parameters, samples


def test_cut_wav_frames_preserves_header_samples_and_timing(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "parts" / "clip.wav"
    samples = list(range(1_000))
    _write_wav(source, samples)

    clip = cut_wav_frames(
        source,
        destination,
        start_frame=125,
        end_frame=875,
    )
    parameters, output_samples = _read_samples(destination)

    assert parameters.nchannels == CHANNEL_COUNT
    assert parameters.sampwidth == SAMPLE_WIDTH_BYTES
    assert parameters.framerate == SAMPLE_RATE
    assert parameters.nframes == 750
    assert output_samples == samples[125:875]
    assert clip.frame_count == 750
    assert clip.relative_start_ms == frame_to_milliseconds(125)
    assert clip.relative_end_ms == frame_to_milliseconds(875)
    assert clip.duration_ms == clip.relative_end_ms - clip.relative_start_ms


def test_cut_wav_frames_streams_ranges_larger_than_one_copy_block(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "clip.wav"
    frame_count = COPY_BLOCK_FRAMES * 2 + 17
    samples = [index % 30_000 for index in range(frame_count)]
    _write_wav(source, samples)

    clip = cut_wav_frames(
        source,
        destination,
        start_frame=0,
        end_frame=frame_count,
    )
    _, output_samples = _read_samples(destination)

    assert clip.frame_count == frame_count
    assert output_samples == samples


@pytest.mark.parametrize(
    ("start_frame", "end_frame"),
    [(-1, 10), (10, 10), (11, 10), (0, 101)],
)
def test_invalid_ranges_fail_without_leaving_output(
    tmp_path: Path,
    start_frame: int,
    end_frame: int,
) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "clip.wav"
    _write_wav(source, list(range(100)))

    with pytest.raises(WavCutError):
        cut_wav_frames(
            source,
            destination,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    assert not destination.exists()


def test_source_cannot_be_used_as_the_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, list(range(100)))

    with pytest.raises(WavCutError, match="source and destination paths must differ"):
        cut_wav_frames(
            source,
            source,
            start_frame=0,
            end_frame=10,
        )

    _, samples = _read_samples(source)
    assert samples == list(range(100))


def test_frame_millisecond_conversion_is_integer_and_deterministic() -> None:
    assert frame_to_milliseconds(0) == 0
    assert frame_to_milliseconds(7) == 0
    assert frame_to_milliseconds(8) == 1
    assert frame_to_milliseconds(SAMPLE_RATE) == 1_000
    assert milliseconds_to_frames(1) == 16
    assert milliseconds_to_frames(1_000) == SAMPLE_RATE
