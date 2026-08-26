from __future__ import annotations

import base64
import io
import math
import wave
from collections.abc import Callable

from voice_pipeline_chunk_contracts import (
    AlignedTextUnit,
    build_segment_word_alignment,
    parse_text_with_audio_tags,
)

_LANGUAGES = {
    "yue": "Cantonese",
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
}


class Qwen3SegmentAligner:
    """Lazily loaded Qwen3 aligner for isolated generated WAV segments."""

    def __init__(self, policy, model_factory: Callable | None = None):
        self.policy = policy
        self._model_factory = model_factory
        self._model = None
        self._torch = None

    def align(
        self, audio: bytes, *, text_with_audio_tags: str, language: str
    ) -> list[dict[str, object]]:
        duration_ms = _wav_duration_ms(audio)
        tagged = parse_text_with_audio_tags(text_with_audio_tags)
        if language not in _LANGUAGES:
            raise ValueError("forced-alignment language is unsupported")
        if not tagged.text:
            units: list[AlignedTextUnit] = []
        else:
            model = self._load()
            encoded = "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")
            results = model.align(
                audio=encoded,
                text=tagged.text,
                language=_LANGUAGES[language],
            )
            if not isinstance(results, list) or len(results) != 1:
                raise RuntimeError("Qwen3 forced aligner returned an invalid result")
            timestamp_tolerance_ms = _timestamp_tolerance_ms(model)
            units = [
                _aligned_unit(item, duration_ms, timestamp_tolerance_ms)
                for item in results[0]
            ]
        return build_segment_word_alignment(
            text_with_audio_tags,
            units,
            duration_ms=duration_ms,
        )

    def _load(self):
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            self._model = self._model_factory(self.policy)
            return self._model

        import torch
        from huggingface_hub import snapshot_download
        from qwen_asr import Qwen3ForcedAligner

        if self.policy.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for forced alignment but is unavailable"
            )
        model_path = snapshot_download(
            repo_id=self.policy.repo_id,
            revision=self.policy.revision,
        )
        self._model = Qwen3ForcedAligner.from_pretrained(
            model_path,
            dtype=getattr(torch, self.policy.dtype),
            device_map=self.policy.device,
        )
        self._torch = torch
        return self._model

    def close(self):
        if self._model is not None:
            model = getattr(self._model, "model", None)
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
            self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _aligned_unit(
    item: object, duration_ms: int, timestamp_tolerance_ms: int
) -> AlignedTextUnit:
    text = getattr(item, "text", None)
    start_seconds = getattr(item, "start_time", None)
    end_seconds = getattr(item, "end_time", None)
    if (
        not isinstance(text, str)
        or not text
        or isinstance(start_seconds, bool)
        or not isinstance(start_seconds, int | float)
        or isinstance(end_seconds, bool)
        or not isinstance(end_seconds, int | float)
        or not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
    ):
        raise RuntimeError("Qwen3 forced aligner returned a malformed item")
    start_ms = round(start_seconds * 1000)
    end_ms = round(end_seconds * 1000)
    if (
        not 0 <= start_ms <= end_ms
        or start_ms > duration_ms + timestamp_tolerance_ms
        or end_ms > duration_ms + timestamp_tolerance_ms
    ):
        raise RuntimeError(
            "Qwen3 forced aligner timestamp is outside the segment: "
            f"text={text!r}, start_ms={start_ms}, end_ms={end_ms}, "
            f"duration_ms={duration_ms}, tolerance_ms={timestamp_tolerance_ms}"
        )
    return AlignedTextUnit(
        text,
        min(start_ms, duration_ms),
        min(end_ms, duration_ms),
    )


def _timestamp_tolerance_ms(model: object) -> int:
    value = getattr(model, "timestamp_segment_time", 80)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RuntimeError("Qwen3 forced aligner timestamp resolution is invalid")
    return math.ceil(value)


def _wav_duration_ms(payload: bytes) -> int:
    try:
        with wave.open(io.BytesIO(payload), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getcomptype() != "NONE"
                or reader.getframerate() <= 0
                or reader.getnframes() <= 0
            ):
                raise ValueError("forced-alignment WAV format is invalid")
            return round(reader.getnframes() * 1000 / reader.getframerate())
    except (EOFError, wave.Error) as exc:
        raise ValueError("forced-alignment input is not a valid WAV") from exc
