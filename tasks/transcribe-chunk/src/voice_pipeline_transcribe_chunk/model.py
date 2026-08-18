from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from . import numpy_compat  # noqa: F401 - must run before importing NeMo
from .utterances import DecodedWord


class ParakeetModel:
    """Lazily loaded, process-local adapter for the pinned Parakeet artifact."""

    def __init__(self, policy, token: str | None = None):
        self.policy = policy
        self.token = token
        self._model = None
        self._torch = None

    def _artifact(self) -> Path:
        path = Path(
            hf_hub_download(
                repo_id=self.policy.model.repo_id,
                filename=self.policy.model.filename,
                revision=self.policy.model.revision,
                token=self.token,
            )
        )
        if path.stat().st_size != self.policy.model.size_bytes:
            raise RuntimeError("model artifact size mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.policy.model.sha256:
            raise RuntimeError("model artifact checksum mismatch")
        return path

    def _load(self):
        if self._model is not None:
            return self._model
        import nemo.collections.asr as nemo_asr
        import torch
        from omegaconf import OmegaConf, open_dict

        if self.policy.model.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = (
            "cuda"
            if self.policy.model.device == "cuda"
            or (self.policy.model.device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        model = nemo_asr.models.ASRModel.restore_from(str(self._artifact()))
        model = model.to(device)
        model.eval()
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[
                self.policy.decoder.attention_left,
                self.policy.decoder.attention_right,
            ],
        )
        decoding = deepcopy(model.cfg.decoding)
        with open_dict(decoding):
            decoding.preserve_alignments = True
            decoding.confidence_cfg = OmegaConf.create(
                {
                    "preserve_frame_confidence": False,
                    "preserve_token_confidence": True,
                    "preserve_word_confidence": True,
                    "exclude_blank": True,
                    "tdt_include_duration": False,
                    "aggregation": "min",
                    "method_cfg": {"name": "max_prob"},
                }
            )
        model.change_decoding_strategy(decoding)
        self._model = model
        self._torch = torch
        return model

    def transcribe(self, audio: np.ndarray) -> list[DecodedWord]:
        if audio.ndim != 1 or audio.dtype != np.float32 or not np.isfinite(audio).all():
            raise ValueError("ASR slice must be finite mono float32 audio")
        if audio.size == 0:
            return []
        model = self._load()
        with self._torch.inference_mode():
            output = model.transcribe([audio], timestamps=True)
        if isinstance(output, tuple):
            if len(output) != 2:
                raise RuntimeError("unexpected Parakeet output shape")
            output = output[0]
        if not isinstance(output, list) or len(output) != 1:
            raise RuntimeError("unexpected Parakeet output shape")
        result = output[0]
        if isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError("unexpected nested Parakeet output shape")
            result = result[0]
        timestep = getattr(result, "timestep", None)
        if not isinstance(timestep, dict) or not isinstance(timestep.get("word"), list):
            raise TypeError("Parakeet word timestamps are missing")
        raw_words = timestep["word"]
        confidences = list(getattr(result, "word_confidence", None) or [])
        if len(confidences) != len(raw_words):
            raise RuntimeError("Parakeet word confidence count mismatch")
        decoded = []
        for raw, confidence in zip(raw_words, confidences, strict=True):
            if not isinstance(raw, dict) or not {"word", "start", "end"} <= set(raw):
                raise RuntimeError("Parakeet word entry is malformed")
            score = float(confidence)
            start = float(raw["start"])
            end = float(raw["end"])
            if not all(math.isfinite(value) for value in (score, start, end)):
                raise RuntimeError("Parakeet returned a non-finite value")
            decoded.append(DecodedWord(str(raw["word"]), start, end, score))
        return decoded

    def close(self):
        if self._model is not None:
            self._model.to("cpu")
            self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
