from __future__ import annotations

import multiprocessing
import os
import time
import wave

import pytest

from voice_pipeline_ingest_api.services.audio_normalizer import (
    PYDUB_TIMEOUT_SECONDS,
    AudioNormalizationError,
    AudioNormalizationTimeout,
    AudioNormalizer,
)


def test_production_pydub_timeout_is_thirty_seconds() -> None:
    assert PYDUB_TIMEOUT_SECONDS == 30.0


def _write_partial_output_and_wait(source_path: str, destination_path: str) -> None:
    del source_path
    with open(destination_path, "wb") as output:
        output.write(b"partial")
        output.flush()
        os.fsync(output.fileno())
    time.sleep(30)


def _write_invalid_output(source_path: str, destination_path: str) -> None:
    del source_path
    with open(destination_path, "wb") as output:
        output.write(b"invalid")


def _write_source_wav(path) -> None:
    channels = 2
    sample_width = 2
    sample_rate = 44_100
    frame_count = 4_410
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frame_count * channels * sample_width)


def test_audio_normalizer_creates_valid_contract_wav(tmp_path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "normalized.wav"
    _write_source_wav(source)

    metadata = AudioNormalizer().normalize(source, destination)

    assert metadata.duration_ms == 100
    with wave.open(str(destination), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getcomptype() == "NONE"


def test_audio_normalizer_interrupts_timeout_and_removes_output(tmp_path) -> None:
    source = tmp_path / "source.audio"
    destination = tmp_path / "normalized.wav"
    source.write_bytes(b"source")
    child_pids_before = {child.pid for child in multiprocessing.active_children()}
    normalizer = AudioNormalizer(
        operation=_write_partial_output_and_wait,
        timeout_seconds=0.1,
        termination_grace_seconds=0.1,
    )

    with pytest.raises(AudioNormalizationTimeout):
        normalizer.normalize(source, destination)

    child_pids_after = {child.pid for child in multiprocessing.active_children()}
    assert child_pids_after == child_pids_before
    assert not destination.exists()


def test_audio_normalizer_removes_invalid_worker_output(tmp_path) -> None:
    source = tmp_path / "source.audio"
    destination = tmp_path / "normalized.wav"
    source.write_bytes(b"source")
    normalizer = AudioNormalizer(operation=_write_invalid_output)

    with pytest.raises(AudioNormalizationError, match="invalid WAV"):
        normalizer.normalize(source, destination)

    assert not destination.exists()
