from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .audio import Audio, resample
from .errors import ScoringError

MODEL_ID = "microsoft/wavlm-base-plus-sv"
MODEL_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 10
MINIMUM_SECONDS = 1


class SpeakerSimilarityScorer:
    def __init__(self, device: str = "auto", model_cache: Path | None = None) -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        if device == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise ScoringError("cuda_unavailable")
        else:
            selected = device
        self.device = torch.device(selected)
        cache_dir = str(model_cache / "huggingface") if model_cache else None
        self.extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir
        )
        self.model = (
            WavLMForXVector.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir
            )
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _windows(samples: np.ndarray) -> list[np.ndarray]:
        maximum = WINDOW_SECONDS * SAMPLE_RATE
        minimum = MINIMUM_SECONDS * SAMPLE_RATE
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

    def embedding(self, audio: Audio | tuple[np.ndarray, int]) -> np.ndarray:
        import torch

        samples, rate = (
            (audio.samples, audio.sample_rate_hz) if isinstance(audio, Audio) else audio
        )
        samples = resample(samples, rate, SAMPLE_RATE)
        embeddings: list[np.ndarray] = []
        weights: list[float] = []
        try:
            for window in self._windows(samples):
                inputs = self.extractor(
                    window,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    vector = self.model(
                        **{key: value.to(self.device) for key, value in inputs.items()}
                    ).embeddings
                    vector = torch.nn.functional.normalize(vector, dim=-1)
                embeddings.append(vector[0].cpu().numpy())
                weights.append(window.size / SAMPLE_RATE)
        except ScoringError:
            raise
        except Exception as exc:
            raise ScoringError("speaker_embedding_failed") from exc
        pooled = np.average(np.stack(embeddings), axis=0, weights=np.asarray(weights))
        norm = float(np.linalg.norm(pooled))
        if not math.isfinite(norm) or norm <= 0:
            raise ScoringError("speaker_embedding_non_finite")
        return (pooled / norm).astype(np.float32)

    @staticmethod
    def similarity(output: np.ndarray, reference: np.ndarray) -> float:
        value = float(np.dot(output, reference))
        if not math.isfinite(value):
            raise ScoringError("speaker_similarity_non_finite")
        return value

    def manifest(self) -> dict[str, object]:
        return {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": str(self.device),
            "sample_rate_hz": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "pooling": "duration-weighted-mean-then-l2",
        }

    def close(self) -> None:
        self.model = None
