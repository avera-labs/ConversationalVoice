from __future__ import annotations

import logging
import wave
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .config import EnvironmentSettings, VadPolicy
from .vad_artifact import (
    VadArtifactDocument,
    build_vad_artifact,
    write_vad_artifact,
)
from .wav_io import SAMPLE_RATE, SAMPLE_WIDTH_BYTES
from .windowing import FrameSpan, normalize_segments

DeviceName = Literal["cpu", "cuda", "mps"]
DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
PipelineProvider = Callable[[], Any]
WaveformLoader = Callable[[Path], tuple[Any, int]]
logger = logging.getLogger(__name__)

_HF_ACCESS_ERROR_NAMES = frozenset(
    {
        "BadRequestError",
        "GatedRepoError",
        "RepositoryNotFoundError",
    }
)


class VadError(RuntimeError):
    """Base class for safe VAD wrapper failures."""


class VadDeviceError(VadError):
    """Raised when the configured inference device is unavailable."""


class VadModelLoadError(VadError):
    """Raised when the gated segmentation model cannot be loaded."""


class VadInputError(VadError):
    """Raised when the normalized input WAV cannot be read for inference."""


class VadInferenceError(VadError):
    """Raised when model inference or output extraction fails."""


class VadArtifactError(VadError):
    """Raised when normalized VAD output cannot be written as JSON."""


@dataclass(frozen=True, slots=True)
class VadResult:
    """Normalized speech activity aligned to the source WAV frame timeline."""

    model: str
    audio_frame_count: int
    segments: tuple[FrameSpan, ...]

    def artifact_document(self) -> VadArtifactDocument:
        return build_vad_artifact(
            model=self.model,
            audio_frame_count=self.audio_frame_count,
            segments=list(self.segments),
        )


def resolve_device(
    preference: DevicePreference,
    *,
    torch_module: Any | None = None,
) -> DeviceName:
    """Resolve a reviewed device preference without silent explicit fallback."""

    if preference not in ("auto", "cpu", "cuda", "mps"):
        raise VadDeviceError("The configured VAD device is invalid.")
    if preference == "cpu":
        return "cpu"

    try:
        if torch_module is None:
            import torch as torch_module

        cuda_available = bool(torch_module.cuda.is_available())
        mps_backend = getattr(torch_module.backends, "mps", None)
        mps_available = bool(
            mps_backend is not None and mps_backend.is_available()
        )
    except Exception as exc:
        raise VadDeviceError("Unable to inspect VAD inference devices.") from exc

    if preference == "cuda":
        if not cuda_available:
            raise VadDeviceError("The configured CUDA device is unavailable.")
        return "cuda"
    if preference == "mps":
        if not mps_available:
            raise VadDeviceError("The configured MPS device is unavailable.")
        return "mps"
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _load_pipeline_uncached(
    model_name: str,
    token: str,
    device: DeviceName,
) -> Any:
    """Import and construct the model runtime only on first inference."""

    import torch
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    segmentation = Model.from_pretrained(model_name, token=token)
    pipeline = VoiceActivityDetection(segmentation=segmentation)
    pipeline.instantiate(
        {
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }
    )
    if device != "cpu":
        pipeline.to(torch.device(device))
    return pipeline


@lru_cache(maxsize=4)
def _cached_pipeline(
    model_name: str,
    token: str,
    device: DeviceName,
) -> Any:
    """Cache one pipeline per reviewed model and device within the process."""

    return _load_pipeline_uncached(model_name, token, device)


def clear_vad_pipeline_cache() -> None:
    """Clear process-local model state for controlled tests and shutdown."""

    _cached_pipeline.cache_clear()


def _is_hugging_face_access_error(error: BaseException) -> bool:
    """Recognize access failures without rendering provider diagnostics."""

    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if type(current).__name__ in _HF_ACCESS_ERROR_NAMES:
            return True
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) in (401, 403):
            return True
        current = current.__cause__ or current.__context__
    return False


def _read_pcm_waveform(
    path: Path,
    *,
    torch_module: Any | None = None,
) -> tuple[Any, int]:
    """Read the task's normalized PCM dependency into a pyannote payload."""

    # Task input dependency: non-empty 16 kHz mono 16-bit PCM WAV.
    # Ingest owns format validation; this reader only checks that all expected
    # PCM frames required for inference can be consumed.
    try:
        with wave.open(str(path), "rb") as reader:
            frame_count = reader.getnframes()
            pcm_bytes = reader.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise VadInputError("Unable to read the normalized WAV.") from exc

    if frame_count <= 0:
        raise VadInputError("The normalized WAV is empty.")
    expected_bytes = frame_count * SAMPLE_WIDTH_BYTES
    if len(pcm_bytes) != expected_bytes:
        raise VadInputError("The normalized WAV ended before inference input.")

    try:
        if torch_module is None:
            import torch as torch_module

        waveform = torch_module.frombuffer(
            bytearray(pcm_bytes),
            dtype=torch_module.int16,
            count=frame_count,
        )
        waveform = waveform.to(dtype=torch_module.float32)
        waveform = waveform.div_(32_768.0).unsqueeze(0).contiguous()
    except Exception as exc:
        raise VadInputError("Unable to prepare the normalized WAV.") from exc
    return waveform, frame_count


class PyannoteVad:
    """Lazy process-cached pyannote segmentation VAD wrapper."""

    def __init__(
        self,
        *,
        model_name: str,
        device: DeviceName,
        pipeline_provider: PipelineProvider,
        waveform_loader: WaveformLoader = _read_pcm_waveform,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self._model_name = model_name
        self._device = device
        self._pipeline_provider = pipeline_provider
        self._waveform_loader = waveform_loader
        self._pipeline: Any | None = None

    @classmethod
    def create(
        cls,
        policy: VadPolicy,
        environment: EnvironmentSettings,
    ) -> PyannoteVad:
        device = resolve_device(policy.device)
        token = environment.hf_token.get_secret_value()
        return cls(
            model_name=policy.model,
            device=device,
            pipeline_provider=lambda: _cached_pipeline(
                policy.model,
                token,
                device,
            ),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> DeviceName:
        return self._device

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            logger.info(
                (
                    "Loading VAD model model=%s device=%s; "
                    "Hugging Face authentication is configured."
                ),
                self._model_name,
                self._device,
            )
            try:
                self._pipeline = self._pipeline_provider()
            except Exception as exc:
                if _is_hugging_face_access_error(exc):
                    logger.error(
                        (
                            "Hugging Face denied access to VAD model model=%s. "
                            "Verify that HF_TOKEN is valid and its account has "
                            "accepted the model access conditions."
                        ),
                        self._model_name,
                    )
                else:
                    logger.error(
                        (
                            "Unable to load VAD model model=%s. If the model is "
                            "gated, verify HF_TOKEN and acceptance of its access "
                            "conditions."
                        ),
                        self._model_name,
                    )
                raise VadModelLoadError("Unable to load the VAD model.") from exc
            if self._pipeline is None:
                logger.error(
                    (
                        "VAD model loader returned no model model=%s. Verify "
                        "HF_TOKEN and acceptance of the model access conditions."
                    ),
                    self._model_name,
                )
                raise VadModelLoadError("Unable to load the VAD model.")
            logger.info(
                "VAD model loaded model=%s device=%s.",
                self._model_name,
                self._device,
            )
        return self._pipeline

    def run(self, audio_path: Path) -> VadResult:
        try:
            waveform, audio_frame_count = self._waveform_loader(audio_path)
        except VadInputError:
            raise
        except Exception as exc:
            raise VadInputError("Unable to read the normalized WAV.") from exc

        pipeline = self._get_pipeline()
        try:
            annotation = pipeline(
                {
                    "waveform": waveform,
                    "sample_rate": SAMPLE_RATE,
                }
            )
            raw_segments = [
                (float(segment.start), float(segment.end))
                for segment in annotation.itersegments()
            ]
            normalized = normalize_segments(
                raw_segments,
                audio_frame_count=audio_frame_count,
            )
        except Exception as exc:
            raise VadInferenceError("VAD inference failed.") from exc

        return VadResult(
            model=self._model_name,
            audio_frame_count=audio_frame_count,
            segments=tuple(normalized),
        )

    def run_and_write(self, audio_path: Path, output_path: Path) -> VadResult:
        result = self.run(audio_path)
        try:
            write_vad_artifact(output_path, result.artifact_document())
        except Exception as exc:
            raise VadArtifactError("Unable to write the VAD artifact.") from exc
        return result
