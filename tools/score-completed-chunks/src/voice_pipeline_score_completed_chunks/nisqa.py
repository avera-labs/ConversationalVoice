from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .errors import ScoringError
from .model_files import NISQA_MODEL, ensure_model_file

MAXIMUM_WINDOW_SECONDS = 50
MINIMUM_WINDOW_SECONDS = 1
METRIC_NAMES = (
    "nisqa_mos",
    "nisqa_noisiness",
    "nisqa_discontinuity",
    "nisqa_coloration",
    "nisqa_loudness",
)


class NisqaScorer:
    def __init__(self, model_cache: Path) -> None:
        directory = model_cache / "nisqa"
        self.model_path = ensure_model_file(NISQA_MODEL, directory / NISQA_MODEL.name)

        # TorchMetrics otherwise hard-codes ~/.torchmetrics/NISQA. Point its
        # loader at the verified file in this tool's configured model cache.
        from torchmetrics.functional.audio import nisqa as torchmetrics_nisqa

        torchmetrics_nisqa.NISQA_DIR = str(directory)
        torchmetrics_nisqa._load_nisqa_model.cache_clear()

    @staticmethod
    def _windows(samples: np.ndarray, sample_rate_hz: int) -> list[np.ndarray]:
        if sample_rate_hz <= 0 or samples.size == 0:
            raise ScoringError("nisqa_empty_audio")
        maximum = MAXIMUM_WINDOW_SECONDS * sample_rate_hz
        minimum = MINIMUM_WINDOW_SECONDS * sample_rate_hz
        windows = [
            samples[start : start + maximum]
            for start in range(0, samples.size, maximum)
        ]
        if len(windows) > 1 and windows[-1].size < minimum:
            windows[-2] = np.concatenate((windows[-2], windows[-1]))
            windows.pop()
        if not windows or windows[0].size < minimum:
            raise ScoringError("insufficient_active_speech")
        return windows

    @staticmethod
    def _score_window(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        import torch
        from torchmetrics.functional.audio.nisqa import (
            non_intrusive_speech_quality_assessment,
        )

        values = non_intrusive_speech_quality_assessment(
            torch.as_tensor(samples, dtype=torch.float32), sample_rate_hz
        ).reshape(-1)
        if values.numel() != len(METRIC_NAMES):
            raise ScoringError("nisqa_invalid_output")
        return np.asarray(values.cpu().tolist(), dtype=np.float64)

    def score(
        self, samples: np.ndarray, sample_rate_hz: int
    ) -> dict[str, float | int]:
        try:
            windows = self._windows(samples, sample_rate_hz)
            values = np.stack(
                [self._score_window(window, sample_rate_hz) for window in windows]
            )
        except Exception as exc:
            if isinstance(exc, ScoringError):
                raise
            raise ScoringError("nisqa_failed") from exc
        result = np.average(
            values,
            axis=0,
            weights=np.asarray([window.size for window in windows], dtype=np.float64),
        )
        if not all(math.isfinite(float(item)) for item in result):
            raise ScoringError("nisqa_non_finite")
        return {
            **{
                name: float(value)
                for name, value in zip(METRIC_NAMES, result, strict=True)
            },
            "nisqa_window_count": len(windows),
        }

    def manifest(self) -> dict[str, object]:
        return {
            "implementation": "torchmetrics.functional.audio.nisqa",
            "upstream_commit": "fe84f0f252abec382b24367d5b22498a7ce34dbb",
            "maximum_window_seconds": MAXIMUM_WINDOW_SECONDS,
            "window_aggregation": "duration-weighted-mean",
            "weights": {
                "name": NISQA_MODEL.name,
                "sha256": NISQA_MODEL.sha256,
                "size_bytes": NISQA_MODEL.size_bytes,
            },
        }
