from __future__ import annotations

import json
import os
import platform
import time
import wave
from pathlib import Path

import keras
import librosa
import pytest
import tensorflow as tf

from voice_pipeline_quality_filter_audio_part.certification import detect_execution_target
from voice_pipeline_quality_filter_audio_part.config import load_settings
from voice_pipeline_quality_filter_audio_part.music import KerasMusicDetector
from voice_pipeline_quality_filter_audio_part.wav_io import read_normalized_wav

pytestmark = pytest.mark.model_smoke
if os.environ.get("RUN_MODEL_SMOKE_TEST") != "1":
    pytest.skip(
        "Set RUN_MODEL_SMOKE_TEST=1 through tests/smoke/run.sh.",
        allow_module_level=True,
    )


def settings_environment() -> dict[str, str]:
    environment = {
        "DATABASE_URL": "postgresql://unused/unused",
        "CELERY_BROKER_URL": "redis://unused/0",
        "S3_BUCKET": "unused",
        "S3_REGION": "unused",
    }
    for name in ("MUSIC_MODEL_CACHE_DIR", "QUALITY_FILTER_CONFIG_FILE"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        frames = reader.getnframes()
    assert frames % 16 == 0
    return frames // 16


def test_real_model_checksum_load_prediction_and_process_cache() -> None:
    settings = load_settings(settings_environment())
    target = detect_execution_target()
    audio_path = Path(os.environ["MODEL_SMOKE_WAV"]).resolve()
    assert audio_path.is_file()
    duration_ms = wav_duration_ms(audio_path)
    audio = read_normalized_wav(audio_path, expected_duration_ms=duration_ms)
    expected = tuple(tuple(item) for item in json.loads(os.environ["EXPECTED_MUSIC_INTERVALS_JSON"]))
    detector = KerasMusicDetector(
        cache_dir=settings.environment.music_model_cache_dir,
        music_policy=settings.policy.music,
        quality_policy=settings.policy.quality,
    )
    detector.validate_artifacts()
    started = time.perf_counter()
    try:
        first = detector.detect(
            audio.waveform, sample_rate=audio.sample_rate, duration_ms=duration_ms
        )
        loaded_model_identity = id(detector._model)
        second = detector.detect(
            audio.waveform, sample_rate=audio.sample_rate, duration_ms=duration_ms
        )
        elapsed = time.perf_counter() - started
        assert id(detector._model) == loaded_model_identity
    finally:
        detector.close()
    actual = tuple((item.start_ms, item.end_ms) for item in first)
    assert actual == expected
    assert second == first
    report = {
        "test": "short_wav",
        "architecture": platform.machine(),
        "device": target.device,
        "accelerator": target.accelerator,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "librosa": librosa.__version__,
        "model": settings.policy.music.model_name,
        "audio_duration_ms": duration_ms,
        "music_intervals": actual,
        "elapsed_seconds": round(elapsed, 3),
    }
    Path(os.environ["COMPATIBILITY_REPORT_PATH"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
