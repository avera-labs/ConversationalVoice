from __future__ import annotations

import json
import os
import platform
import resource
import time
import wave
from pathlib import Path

import psutil
import pytest
import torch
import torchaudio

from voice_pipeline_diarize_audio_part.artifact import build_artifact
from voice_pipeline_diarize_audio_part.certification import (
    validate_certification_target,
)
from voice_pipeline_diarize_audio_part.config import load_settings
from voice_pipeline_diarize_audio_part.diarization import DiarizationEngine

pytestmark = pytest.mark.capacity
if os.environ.get("RUN_CAPACITY_TESTS") != "1":
    pytest.skip(
        "Set RUN_CAPACITY_TESTS=1 through tests/capacity/run.sh.",
        allow_module_level=True,
    )


def assert_target() -> tuple[str, str]:
    target = os.environ["CERTIFICATION_TARGET"]
    architecture = platform.machine().lower()
    accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    validate_certification_target(
        target=target,
        architecture=architecture,
        cuda_available=torch.cuda.is_available(),
        accelerator=accelerator,
    )
    return target, accelerator


def write_long_wav(source: Path, path: Path) -> None:
    with wave.open(str(source), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        frames = reader.readframes(reader.getnframes())
    assert frames
    target_bytes = 900 * 16000 * 2
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        remaining = target_bytes
        while remaining:
            block = frames[:remaining]
            writer.writeframesraw(block)
            remaining -= len(block)


def test_model_capacity_on_900_seconds(tmp_path: Path) -> None:
    target, accelerator = assert_target()
    settings = load_settings(
        {
            "DATABASE_URL": "postgresql://unused/unused",
            "CELERY_BROKER_URL": "redis://unused/0",
            "S3_BUCKET": "unused",
            "S3_REGION": "unused",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        }
    )
    policy = settings.policy.diarization
    audio_path = tmp_path / "capacity.wav"
    write_long_wav(Path(os.environ["CAPACITY_SOURCE_WAV"]).resolve(), audio_path)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    engine = DiarizationEngine(
        model=policy.model,
        requested_device="cuda",
    )
    started = time.perf_counter()
    try:
        result = engine.infer(audio_path)
        artifact = build_artifact(result.turns, model=policy.model, duration_ms=900_000)
        elapsed = time.perf_counter() - started
        report = {
            "test": "900_second_wav",
            "target": target,
            "architecture": platform.machine(),
            "accelerator": accelerator,
            "accelerator_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "model": policy.model,
            "input_bytes": audio_path.stat().st_size,
            "segment_count": len(artifact.segments),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "rss_growth_bytes": max(0, process.memory_info().rss - rss_before),
            "temporary_disk_bytes": audio_path.stat().st_size,
            "elapsed_seconds": round(elapsed, 3),
        }
        Path(os.environ["CAPACITY_REPORT_PATH"]).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        engine.close()
    assert result.device == "cuda"
    assert all(
        0 <= segment.start < segment.end <= 900.0 for segment in artifact.segments
    )
