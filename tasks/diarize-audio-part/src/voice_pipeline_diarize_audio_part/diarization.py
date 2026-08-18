"""Lazy DiariZen pipeline and annotation extraction."""

from __future__ import annotations

import gc
import os
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact import RawTurn


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device: str
    accelerator: str | None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    turns: tuple[RawTurn, ...]
    device: str
    accelerator: str | None
    model_cache_hit: bool


def resolve_device(requested: str, torch_module: Any) -> DeviceInfo:
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("unsupported diarization device")
    if requested == "auto":
        if torch_module.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    accelerator = torch_module.cuda.get_device_name(0) if requested == "cuda" else None
    return DeviceInfo(device=requested, accelerator=accelerator)


def extract_turns(output: Any) -> tuple[RawTurn, ...]:
    annotation = getattr(output, "speaker_diarization", output)
    iterator = getattr(annotation, "itertracks", None)
    if not callable(iterator):
        raise TypeError("model output does not provide diarization tracks")
    extracted: list[RawTurn] = []
    for segment, _, speaker in iterator(yield_label=True):
        extracted.append(
            RawTurn(
                start=float(segment.start),
                end=float(segment.end),
                speaker_label=str(speaker),
            )
        )
    return tuple(extracted)


PipelineLoader = Callable[[str, str], Any]


@contextmanager
def discard_third_party_console_output():
    """Prevent model libraries from printing cache paths or remote details."""
    with (
        open(os.devnull, "w", encoding="utf-8") as sink,
        redirect_stdout(sink),
        redirect_stderr(sink),
    ):
        yield


def apply_torch_26_checkpoint_compatibility(
    torch_module: Any,
    *,
    pyannote_types: tuple[type, ...] | None = None,
) -> None:
    """Allow only metadata types present in the supported checkpoint format."""
    if pyannote_types is None:
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        pyannote_types = (Problem, Resolution, Specifications)
    torch_module.serialization.add_safe_globals(
        [torch_module.torch_version.TorchVersion, *pyannote_types]
    )


@contextmanager
def diarizen_initialization_device(torch_module: Any, device: str):
    """Make DiariZen select the requested device while it constructs its models."""
    if device != "cpu" or not torch_module.cuda.is_available():
        yield
        return

    original_is_available = torch_module.cuda.is_available
    torch_module.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch_module.cuda.is_available = original_is_available


def _load_pipeline(model: str, device: str) -> Any:
    import torch
    from diarizen.pipelines.inference import DiariZenPipeline

    apply_torch_26_checkpoint_compatibility(torch)
    with diarizen_initialization_device(torch, device):
        return DiariZenPipeline.from_pretrained(model)


class DiarizationEngine:
    """Process-local lazy model cache."""

    def __init__(
        self,
        *,
        model: str,
        requested_device: str,
        loader: PipelineLoader = _load_pipeline,
        torch_module: Any | None = None,
    ) -> None:
        self.model = model
        self._loader = loader
        self._torch = torch_module
        self._requested_device = requested_device
        self._pipeline: Any | None = None
        self._device_info: DeviceInfo | None = None
        self._lock = threading.Lock()

    def _get_pipeline(self) -> tuple[Any, DeviceInfo, bool]:
        with self._lock:
            hit = self._pipeline is not None
            if self._pipeline is None:
                if self._torch is None:
                    import torch

                    self._torch = torch
                self._device_info = resolve_device(self._requested_device, self._torch)
                with discard_third_party_console_output():
                    self._pipeline = self._loader(
                        self.model,
                        self._device_info.device,
                    )
            assert self._device_info is not None
            return self._pipeline, self._device_info, hit

    def infer(self, audio_path: Path) -> InferenceResult:
        pipeline, device_info, cache_hit = self._get_pipeline()
        with discard_third_party_console_output():
            output = pipeline(str(audio_path))
        return InferenceResult(
            turns=extract_turns(output),
            device=device_info.device,
            accelerator=device_info.accelerator,
            model_cache_hit=cache_hit,
        )

    def close(self) -> None:
        with self._lock:
            self._pipeline = None
            self._device_info = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def labels_from_tuples(
    items: Iterable[tuple[float, float, str]],
) -> tuple[RawTurn, ...]:
    """Small public helper for deterministic fake-model tests."""
    return tuple(
        RawTurn(start=start, end=end, speaker_label=label)
        for start, end, label in items
    )
