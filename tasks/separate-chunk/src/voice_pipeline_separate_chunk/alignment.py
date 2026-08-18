from __future__ import annotations

import numpy as np

from .errors import QualityRejection, RejectionCode


class WavLMAligner:
    def __init__(self, policy, device="cuda"):
        self.policy = policy
        self.device = device
        self._runtime = None

    def _load(self):
        if self._runtime is not None:
            return self._runtime
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        device = torch.device(
            self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self._runtime = (
            AutoFeatureExtractor.from_pretrained(
                self.policy.repo_id, revision=self.policy.revision
            ),
            WavLMForXVector.from_pretrained(
                self.policy.repo_id, revision=self.policy.revision
            )
            .to(device)
            .eval(),
            device,
        )
        return self._runtime

    def _embedding(self, audio: np.ndarray, sr: int):
        import torch
        import torchaudio

        extractor, model, device = self._load()
        wav = torch.as_tensor(audio, dtype=torch.float32)
        if sr != extractor.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, extractor.sampling_rate)
        values = extractor(
            wav.numpy(), sampling_rate=extractor.sampling_rate, return_tensors="pt"
        )
        with torch.inference_mode():
            emb = model(**{k: v.to(device) for k, v in values.items()}).embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb[0].cpu().numpy()

    def align(self, previous: np.ndarray, current: np.ndarray, sr: int) -> bool:
        rms = np.sqrt(np.mean(previous.astype(np.float64) ** 2, axis=1))
        ref = int(np.argmax(rms))
        reference = self._embedding(previous[ref], sr)
        current_rms = np.sqrt(np.mean(current.astype(np.float64) ** 2, axis=1))
        voiced = [
            i for i, v in enumerate(current_rms) if v >= self.policy.voice_rms_min
        ]
        if not voiced:
            raise QualityRejection(RejectionCode.ALIGNMENT_LOW_CONFIDENCE)
        if len(voiced) == 1:
            matched = voiced[0]
            scores = [0.0, 0.0]
            scores[matched] = 1.0
        else:
            scores = [
                float(np.dot(reference, self._embedding(current[i], sr)))
                for i in range(2)
            ]
            if not np.isfinite(scores).all():
                raise QualityRejection(RejectionCode.ALIGNMENT_LOW_CONFIDENCE)
            matched = int(np.argmax(scores))
            ordered = sorted(scores, reverse=True)
            if (
                ordered[0] < self.policy.similarity_min
                or ordered[0] - ordered[1] < self.policy.margin_min
            ):
                raise QualityRejection(RejectionCode.ALIGNMENT_LOW_CONFIDENCE)
        return matched != ref

    def close(self) -> None:
        self._runtime = None
