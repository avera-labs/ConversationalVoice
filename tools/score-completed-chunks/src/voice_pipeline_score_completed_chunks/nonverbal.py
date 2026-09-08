from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import Audio, resample
from .errors import ScoringError
from .model_files import file_sha256

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
TARGET_SAMPLE_RATE = 16_000
WINDOW_SECONDS = 10

_FAMILY_PATTERNS = {
    "laughter": ("laughter", "giggle", "chuckle", "snicker"),
    "breath": ("breathing", "sigh", "gasp"),
    "cough": ("cough", "throat clearing", "sneeze"),
    "cry": ("crying", "sobbing", "whimper"),
    "other_human_vocalization": (
        "human voice",
        "grunt",
        "groan",
        "hiccup",
        "yawn",
    ),
}


@dataclass(frozen=True, slots=True)
class NonverbalObservation:
    family: str
    label: str
    confidence: float


class NonverbalDetector(Protocol):
    def detect(self, audio: Audio) -> tuple[NonverbalObservation, ...]: ...

    def manifest(self) -> dict[str, object]: ...


class AstNonverbalDetector:
    def __init__(self, *, device: str, model_cache: Path, threshold: float) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        from transformers.utils.hub import cached_file

        selected = (
            "cuda"
            if device == "cuda" or (device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        cache = model_cache / "huggingface"
        self.device = torch.device(selected)
        self.threshold = threshold
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(
                AST_MODEL_ID,
                revision=AST_REVISION,
                cache_dir=cache,
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                AST_MODEL_ID,
                revision=AST_REVISION,
                cache_dir=cache,
                use_safetensors=True,
            ).to(self.device)
            self.model.eval()
            weights = cached_file(
                AST_MODEL_ID,
                "model.safetensors",
                revision=AST_REVISION,
                cache_dir=cache,
            )
        except Exception as exc:
            raise ScoringError("nonverbal_model_load_failed") from exc
        if weights is None:
            raise ScoringError("nonverbal_model_load_failed")
        self.weights_path = Path(weights)
        self.weights_sha256 = file_sha256(self.weights_path)

    @staticmethod
    def _family(label: str) -> str | None:
        lowered = label.casefold()
        for family, patterns in _FAMILY_PATTERNS.items():
            if any(pattern in lowered for pattern in patterns):
                return family
        return None

    def detect(self, audio: Audio) -> tuple[NonverbalObservation, ...]:
        import torch

        samples = resample(audio.samples, audio.sample_rate_hz, TARGET_SAMPLE_RATE)
        if samples.size == 0:
            return ()
        window = WINDOW_SECONDS * TARGET_SAMPLE_RATE
        windows = [
            samples[start : start + window] for start in range(0, samples.size, window)
        ]
        best: dict[str, NonverbalObservation] = {}
        try:
            for values in windows:
                inputs = self.extractor(
                    values,
                    sampling_rate=TARGET_SAMPLE_RATE,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                for index in np.flatnonzero(probabilities >= self.threshold):
                    label = str(self.model.config.id2label[int(index)])
                    family = self._family(label)
                    if family is None:
                        continue
                    observation = NonverbalObservation(
                        family=family,
                        label=label,
                        confidence=float(probabilities[index]),
                    )
                    previous = best.get(family)
                    if previous is None or observation.confidence > previous.confidence:
                        best[family] = observation
        except Exception as exc:
            raise ScoringError("nonverbal_inference_failed") from exc
        return tuple(best[key] for key in sorted(best))

    def manifest(self) -> dict[str, object]:
        return {
            "implementation": "transformers.AutoModelForAudioClassification",
            "model_id": AST_MODEL_ID,
            "revision": AST_REVISION,
            "weights_sha256": self.weights_sha256,
            "license": "bsd-3-clause",
            "sample_rate_hz": TARGET_SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "threshold": self.threshold,
            "device": str(self.device),
        }


class DisabledNonverbalDetector:
    def detect(self, audio: Audio) -> tuple[NonverbalObservation, ...]:
        return ()

    def manifest(self) -> dict[str, object]:
        return {"implementation": "disabled", "claim_available": False}
