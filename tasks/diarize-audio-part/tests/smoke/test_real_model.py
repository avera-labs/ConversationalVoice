from __future__ import annotations

import json
import os
import platform
import time
import wave
from pathlib import Path

import pytest
import torch
import torchaudio

from voice_pipeline_diarize_audio_part.artifact import build_artifact
from voice_pipeline_diarize_audio_part.certification import (
    validate_certification_target,
)
from voice_pipeline_diarize_audio_part.config import load_settings
from voice_pipeline_diarize_audio_part.diarization import DiarizationEngine

pytestmark = pytest.mark.model_smoke
if os.environ.get("RUN_MODEL_SMOKE_TEST") != "1":
    pytest.skip(
        "Set RUN_MODEL_SMOKE_TEST=1 through tests/smoke/run.sh.",
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


def test_real_model_short_wav(tmp_path: Path) -> None:
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
    audio_path = Path(os.environ["MODEL_SMOKE_WAV"]).resolve()
    assert audio_path.is_file()
    with wave.open(str(audio_path), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        duration_ms = round(reader.getnframes() * 1000 / reader.getframerate())
    engine = DiarizationEngine(
        model=policy.model,
        requested_device="cuda",
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = engine.infer(audio_path)
        artifact = build_artifact(
            result.turns, model=policy.model, duration_ms=duration_ms
        )
    finally:
        engine.close()
    report = {
        "test": "short_wav",
        "target": target,
        "architecture": platform.machine(),
        "accelerator": accelerator,
        "accelerator_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "model": policy.model,
        "segment_count": len(artifact.segments),
        "audio_duration_seconds": round(duration_ms / 1000, 3),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    Path(os.environ["COMPATIBILITY_REPORT_PATH"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert result.device == "cuda"
    assert all(
        0 <= segment.start < segment.end <= duration_ms / 1000
        for segment in artifact.segments
    )
