from __future__ import annotations

import math

import numpy as np

from .alignment import DecodedUnit, PUNCTUATION


def _device(requested: str, torch) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"


class ParaformerModel:
    """Lazily loaded Paraformer adapter with token posterior capture."""

    def __init__(self, policy, model_dir: str | None = None):
        self.policy = policy
        self.model_dir = model_dir
        self._auto_model = None
        self._torch = None
        self._captured_logits = None

    def _install_confidence_hook(self, core) -> None:
        """Capture the exact decoder scores used to choose emitted tokens.

        The pinned ModelScope checkpoint loads FunASR's ``SeacoParaformer``.
        That implementation bypasses Paraformer's
        ``cal_decoder_with_predictor`` and decodes through
        ``_seaco_decode_with_ASF`` instead, so hook that method first.  The
        fallback keeps the adapter explicit for compatible Paraformer models.
        """
        method_name = next(
            (
                name
                for name in ("_seaco_decode_with_ASF", "cal_decoder_with_predictor")
                if callable(getattr(core, name, None))
            ),
            None,
        )
        if method_name is None:
            raise RuntimeError("Paraformer decoder confidence hook is unavailable")
        original = getattr(core, method_name)

        def capture(*args, **kwargs):
            output = original(*args, **kwargs)
            scores = output[0] if isinstance(output, tuple) else output
            if not hasattr(scores, "detach"):
                raise RuntimeError("Paraformer decoder scores are unavailable")
            self._captured_logits = scores.detach()
            return output

        setattr(core, method_name, capture)

    def _load(self):
        if self._auto_model is not None:
            return self._auto_model
        import torch
        from funasr import AutoModel

        source = self.model_dir or self.policy.model.repo_id
        auto_model = AutoModel(
            model=source,
            hub="ms",
            model_revision=self.policy.model.revision,
            device=_device(self.policy.model.device, torch),
            disable_update=True,
        )
        core = getattr(auto_model, "model", None)
        if core is None:
            raise RuntimeError("Paraformer decoder confidence hook is unavailable")
        self._install_confidence_hook(core)
        self._auto_model = auto_model
        self._torch = torch
        return auto_model

    def _confidences(self, count: int) -> list[float]:
        logits = self._captured_logits
        core = self._auto_model.model
        if logits is None or logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError("Paraformer token logits are missing")
        probabilities = logits[0].softmax(dim=-1)
        scores, token_ids = probabilities.max(dim=-1)
        excluded = {
            int(getattr(core, name))
            for name in ("blank_id", "sos", "eos")
            if getattr(core, name, None) is not None
        }
        filtered = [
            float(score)
            for score, token_id in zip(scores.tolist(), token_ids.tolist(), strict=True)
            if int(token_id) not in excluded
        ]
        if len(filtered) != count or not all(
            math.isfinite(value) and 0 <= value <= 1 for value in filtered
        ):
            raise RuntimeError("Paraformer confidence count mismatch")
        return filtered

    def transcribe(self, audio: np.ndarray) -> list[DecodedUnit]:
        if audio.ndim != 1 or audio.dtype != np.float32 or not np.isfinite(audio).all():
            raise ValueError("ASR slice must be finite mono float32 audio")
        if audio.size == 0:
            return []
        model = self._load()
        self._captured_logits = None
        with self._torch.inference_mode():
            output = model.generate(
                input=audio,
                fs=16000,
                batch_size_s=300,
                pred_timestamp=True,
            )
        if not isinstance(output, list) or len(output) > 1:
            raise RuntimeError("unexpected Paraformer output shape")
        if not output:
            return []
        result = output[0]
        text = result.get("text")
        timestamps = result.get("timestamp")
        if not isinstance(text, str) or not isinstance(timestamps, list):
            raise RuntimeError("Paraformer text or timestamps are missing")
        units = [char for char in text if not char.isspace()]
        if any(char in PUNCTUATION for char in units) or len(units) != len(timestamps):
            raise RuntimeError("Paraformer character timestamp count mismatch")
        confidences = self._confidences(len(units))
        decoded = []
        for char, stamp, confidence in zip(units, timestamps, confidences, strict=True):
            if (
                not isinstance(stamp, (list, tuple))
                or len(stamp) < 2
                or not all(isinstance(value, int | float) for value in stamp[:2])
            ):
                raise RuntimeError("Paraformer timestamp entry is malformed")
            start = float(stamp[0]) / 1000
            end = float(stamp[1]) / 1000
            if not math.isfinite(start) or not math.isfinite(end):
                raise RuntimeError("Paraformer timestamp is not finite")
            decoded.append(DecodedUnit(char, start, end, confidence))
        return decoded

    def close(self):
        if self._auto_model is not None:
            core = getattr(self._auto_model, "model", None)
            if core is not None and hasattr(core, "to"):
                core.to("cpu")
            self._auto_model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class PunctuationModel:
    def __init__(self, policy, device: str, model_dir: str | None = None):
        self.policy = policy
        self.device = device
        self.model_dir = model_dir
        self._auto_model = None

    def _load(self):
        if self._auto_model is not None:
            return self._auto_model
        import torch
        from funasr import AutoModel

        source = self.model_dir or self.policy.punctuation.repo_id
        self._auto_model = AutoModel(
            model=source,
            hub="ms",
            model_revision=self.policy.punctuation.revision,
            device=_device(self.device, torch),
            disable_update=True,
        )
        return self._auto_model

    def restore(self, text: str) -> str:
        if not text:
            return ""
        output = self._load().generate(input=text)
        if not isinstance(output, list) or len(output) != 1:
            raise RuntimeError("unexpected punctuation output shape")
        restored = output[0].get("text")
        if not isinstance(restored, str) or not restored.strip():
            raise RuntimeError("punctuation output is empty")
        return restored

    def close(self):
        if self._auto_model is not None:
            core = getattr(self._auto_model, "model", None)
            if core is not None and hasattr(core, "to"):
                core.to("cpu")
            self._auto_model = None
