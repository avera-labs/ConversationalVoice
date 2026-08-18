from __future__ import annotations

import json
import os
import platform
import resource
import time
import wave
from pathlib import Path

import pytest
import tensorflow as tf

from voice_pipeline_quality_filter_audio_part.certification import detect_execution_target
from voice_pipeline_quality_filter_audio_part.config import load_settings
from voice_pipeline_quality_filter_audio_part.music import KerasMusicDetector
from voice_pipeline_quality_filter_audio_part.wav_io import read_normalized_wav

pytestmark = pytest.mark.capacity
if os.environ.get("RUN_CAPACITY_TESTS") != "1":
    pytest.skip(
        "Set RUN_CAPACITY_TESTS=1 through tests/capacity/run.sh.",
        allow_module_level=True,
    )


def write_long_wav(source: Path, destination: Path) -> None:
    with wave.open(str(source), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        frames = reader.readframes(reader.getnframes())
    assert frames
    target_bytes = 900 * 16000 * 2
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        remaining = target_bytes
        while remaining:
            block = frames[:remaining]
            writer.writeframesraw(block)
            remaining -= len(block)


def gpu_peak_bytes(target_device: str) -> int | None:
    if target_device != "gpu":
        return None
    try:
        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:
        return None


def test_model_capacity_on_900_seconds_and_two_sequential_calls(tmp_path: Path) -> None:
    environment = {
        "DATABASE_URL": "postgresql://unused/unused",
        "CELERY_BROKER_URL": "redis://unused/0",
        "S3_BUCKET": "unused",
        "S3_REGION": "unused",
    }
    for name in ("MUSIC_MODEL_CACHE_DIR", "QUALITY_FILTER_CONFIG_FILE"):
        if value := os.environ.get(name):
            environment[name] = value
    settings = load_settings(environment)
    target = detect_execution_target()
    audio_path = tmp_path / "capacity.wav"
    write_long_wav(Path(os.environ["CAPACITY_SOURCE_WAV"]).resolve(), audio_path)
    audio = read_normalized_wav(audio_path, expected_duration_ms=900_000)
    detector = KerasMusicDetector(
        cache_dir=settings.environment.music_model_cache_dir,
        music_policy=settings.policy.music,
        quality_policy=settings.policy.quality,
    )
    detector.validate_artifacts()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    started = time.perf_counter()
    try:
        first = detector.detect(
            audio.waveform, sample_rate=audio.sample_rate, duration_ms=900_000
        )
        after_first = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        second = detector.detect(
            audio.waveform, sample_rate=audio.sample_rate, duration_ms=900_000
        )
        after_second = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        elapsed = time.perf_counter() - started
    finally:
        detector.close()
    assert first == second
    assert all(0 <= item.start_ms < item.end_ms <= 900_000 for item in first)
    report = {
        "test": "900_second_wav_two_calls",
        "architecture": platform.machine(),
        "device": target.device,
        "accelerator": target.accelerator,
        "tensorflow": tf.__version__,
        "model": settings.policy.music.model_name,
        "input_bytes": audio_path.stat().st_size,
        "music_interval_count": len(first),
        "peak_rss_bytes": after_second,
        "first_call_rss_growth_bytes": max(0, after_first - rss_before),
        "second_call_peak_growth_bytes": max(0, after_second - after_first),
        "peak_accelerator_memory_bytes": gpu_peak_bytes(target.device),
        "temporary_disk_bytes": audio_path.stat().st_size,
        "elapsed_seconds": round(elapsed, 3),
    }
    Path(os.environ["CAPACITY_REPORT_PATH"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
