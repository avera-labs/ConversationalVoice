from __future__ import annotations

import json
import logging
import math
import sys
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from voice_pipeline_split_raw_audio_into_parts import vad as vad_module
from voice_pipeline_split_raw_audio_into_parts.vad import (
    PyannoteVad,
    VadArtifactError,
    VadDeviceError,
    VadInferenceError,
    VadInputError,
    VadModelLoadError,
    _cached_pipeline,
    _load_pipeline_uncached,
    _read_pcm_waveform,
    clear_vad_pipeline_cache,
    resolve_device,
)
from voice_pipeline_split_raw_audio_into_parts.wav_io import SAMPLE_RATE
from voice_pipeline_split_raw_audio_into_parts.windowing import FrameSpan


class Availability:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


def fake_torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=Availability(cuda),
        backends=SimpleNamespace(mps=Availability(mps)),
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_device_resolution_is_deterministic(
    cuda: bool,
    mps: bool,
    expected: str,
) -> None:
    assert (
        resolve_device("auto", torch_module=fake_torch(cuda=cuda, mps=mps))
        == expected
    )


def test_explicit_device_never_silently_falls_back() -> None:
    runtime = fake_torch(cuda=False, mps=False)

    with pytest.raises(VadDeviceError, match="CUDA device is unavailable"):
        resolve_device("cuda", torch_module=runtime)
    with pytest.raises(VadDeviceError, match="MPS device is unavailable"):
        resolve_device("mps", torch_module=runtime)
    assert resolve_device("cpu", torch_module=runtime) == "cpu"


class FakeAnnotation:
    def __init__(self, segments: list[tuple[float, float]]) -> None:
        self._segments = segments

    def itersegments(self):
        for start, end in self._segments:
            yield SimpleNamespace(start=start, end=end)


class FakePipeline:
    def __init__(
        self,
        segments: list[tuple[float, float]],
        *,
        error: Exception | None = None,
    ) -> None:
        self._segments = segments
        self._error = error
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> FakeAnnotation:
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        return FakeAnnotation(self._segments)


def test_model_is_lazy_reused_and_output_is_normalized() -> None:
    waveform = object()
    pipeline = FakePipeline(
        [
            (2.0, 3.0),
            (0.0, 1.0),
            (0.5, 2.0),
            (math.nan, 3.5),
            (5.0, 6.0),
        ]
    )
    loads: list[None] = []

    def provide_pipeline() -> FakePipeline:
        loads.append(None)
        return pipeline

    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=provide_pipeline,
        waveform_loader=lambda _path: (waveform, SAMPLE_RATE * 4),
    )

    assert loads == []
    first = runner.run(Path("not-opened.wav"))
    second = runner.run(Path("not-opened.wav"))

    assert len(loads) == 1
    assert len(pipeline.payloads) == 2
    assert pipeline.payloads[0] == {
        "waveform": waveform,
        "sample_rate": SAMPLE_RATE,
    }
    assert first == second
    assert first.model == "pyannote/segmentation-3.0"
    assert first.segments == (FrameSpan(0, SAMPLE_RATE * 3),)


def test_process_pipeline_cache_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object()
    loads: list[tuple[str, str, str]] = []

    def load(model: str, token: str, device: str) -> object:
        loads.append((model, token, device))
        return pipeline

    clear_vad_pipeline_cache()
    monkeypatch.setattr(vad_module, "_load_pipeline_uncached", load)
    try:
        first = _cached_pipeline("model", "<token>", "cpu")
        second = _cached_pipeline("model", "<token>", "cpu")
    finally:
        clear_vad_pipeline_cache()

    assert first is pipeline
    assert second is pipeline
    assert loads == [("model", "<token>", "cpu")]


def test_real_loader_contract_passes_token_parameters_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmentation = object()
    model_calls: list[tuple[str, str]] = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model: str, *, token: str) -> object:
            model_calls.append((model, token))
            return segmentation

    class LoadedPipeline:
        def __init__(self, *, segmentation: object) -> None:
            assert segmentation is not None
            self.parameters: dict[str, float] | None = None
            self.device: str | None = None

        def instantiate(self, parameters: dict[str, float]) -> None:
            self.parameters = parameters

        def to(self, device: str) -> None:
            self.device = device

    torch_module = ModuleType("torch")
    torch_module.device = lambda name: f"device:{name}"  # type: ignore[attr-defined]
    pyannote_module = ModuleType("pyannote")
    pyannote_module.__path__ = []  # type: ignore[attr-defined]
    audio_module = ModuleType("pyannote.audio")
    audio_module.__path__ = []  # type: ignore[attr-defined]
    audio_module.Model = FakeModel  # type: ignore[attr-defined]
    pipelines_module = ModuleType("pyannote.audio.pipelines")
    pipelines_module.VoiceActivityDetection = (  # type: ignore[attr-defined]
        LoadedPipeline
    )

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio.pipelines",
        pipelines_module,
    )

    pipeline = _load_pipeline_uncached(
        "pyannote/segmentation-3.0",
        "<token>",
        "cuda",
    )

    assert model_calls == [("pyannote/segmentation-3.0", "<token>")]
    assert pipeline.parameters == {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
    }
    assert pipeline.device == "device:cuda"


def test_model_load_failure_hides_token_and_model_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "hf_private_token_and_response"

    def fail() -> object:
        raise RuntimeError(secret)

    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=fail,
        waveform_loader=lambda _path: (object(), SAMPLE_RATE),
    )

    with caplog.at_level(logging.INFO), pytest.raises(VadModelLoadError) as error:
        runner.run(Path("not-opened.wav"))

    assert str(error.value) == "Unable to load the VAD model."
    assert "Hugging Face authentication is configured" in caplog.text
    assert "If the model is gated" in caplog.text
    assert secret not in str(error.value)
    assert secret not in caplog.text


def test_hugging_face_access_failure_logs_safe_authorization_guidance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "signed provider response"

    class GatedRepoError(RuntimeError):
        pass

    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=lambda: (_ for _ in ()).throw(GatedRepoError(secret)),
        waveform_loader=lambda _path: (object(), SAMPLE_RATE),
    )

    with caplog.at_level(logging.INFO), pytest.raises(VadModelLoadError):
        runner.run(Path("not-opened.wav"))

    assert "Hugging Face denied access" in caplog.text
    assert "accepted the model access conditions" in caplog.text
    assert secret not in caplog.text


def test_model_load_success_logs_model_and_device(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=lambda: FakePipeline([]),
        waveform_loader=lambda _path: (object(), SAMPLE_RATE),
    )

    with caplog.at_level(logging.INFO):
        runner.run(Path("not-opened.wav"))

    assert "VAD model loaded model=pyannote/segmentation-3.0 device=cpu" in caplog.text


def test_inference_failure_hides_model_exception() -> None:
    secret = "private model response"
    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=lambda: FakePipeline([], error=RuntimeError(secret)),
        waveform_loader=lambda _path: (object(), SAMPLE_RATE),
    )

    with pytest.raises(VadInferenceError) as error:
        runner.run(Path("not-opened.wav"))

    assert str(error.value) == "VAD inference failed."
    assert secret not in str(error.value)


def test_run_and_write_persists_normalized_json(tmp_path: Path) -> None:
    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=lambda: FakePipeline([(0.25, 0.75)]),
        waveform_loader=lambda _path: (object(), SAMPLE_RATE),
    )
    output = tmp_path / "vad_segments.json"

    result = runner.run_and_write(Path("not-opened.wav"), output)

    assert result.segments == (FrameSpan(SAMPLE_RATE // 4, SAMPLE_RATE * 3 // 4),)
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "model": "pyannote/segmentation-3.0",
        "audio_duration_ms": 1_000,
        "segments": [
            {
                "index": 0,
                "start_ms": 250,
                "end_ms": 750,
                "duration_ms": 500,
            }
        ],
    }


def test_artifact_write_failure_is_safe(tmp_path: Path) -> None:
    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=lambda: FakePipeline([]),
        waveform_loader=lambda _path: (object(), SAMPLE_RATE),
    )
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(VadArtifactError, match="Unable to write"):
        runner.run_and_write(
            Path("not-opened.wav"),
            parent_file / "vad_segments.json",
        )


class FakeTensor:
    def __init__(self) -> None:
        self.operations: list[tuple[str, Any]] = []

    def to(self, *, dtype: Any) -> FakeTensor:
        self.operations.append(("to", dtype))
        return self

    def div_(self, value: float) -> FakeTensor:
        self.operations.append(("div", value))
        return self

    def unsqueeze(self, dimension: int) -> FakeTensor:
        self.operations.append(("unsqueeze", dimension))
        return self

    def contiguous(self) -> FakeTensor:
        self.operations.append(("contiguous", None))
        return self


class FakeTorchPcm:
    int16 = "int16"
    float32 = "float32"

    def __init__(self) -> None:
        self.tensor = FakeTensor()
        self.buffer: bytes | None = None
        self.count: int | None = None

    def frombuffer(
        self,
        buffer: bytearray,
        *,
        dtype: str,
        count: int,
    ) -> FakeTensor:
        assert dtype == self.int16
        self.buffer = bytes(buffer)
        self.count = count
        return self.tensor


def write_pcm_wav(path: Path, samples: bytes) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(samples)


def test_pcm_reader_uses_fixed_16_khz_mono_int16_dependency(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    samples = b"\x00\x00\xff\x7f\x00\x80"
    write_pcm_wav(audio_path, samples)
    torch_runtime = FakeTorchPcm()

    waveform, frame_count = _read_pcm_waveform(
        audio_path,
        torch_module=torch_runtime,
    )

    assert frame_count == 3
    assert waveform is torch_runtime.tensor
    assert torch_runtime.buffer == samples
    assert torch_runtime.count == 3
    assert torch_runtime.tensor.operations == [
        ("to", "float32"),
        ("div", 32_768.0),
        ("unsqueeze", 0),
        ("contiguous", None),
    ]


def test_empty_pcm_input_fails_before_model_load(tmp_path: Path) -> None:
    audio_path = tmp_path / "empty.wav"
    write_pcm_wav(audio_path, b"")
    loads: list[None] = []
    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=lambda: loads.append(None),
    )

    with pytest.raises(VadInputError, match="WAV is empty"):
        runner.run(audio_path)

    assert loads == []


def test_waveform_loader_failure_is_safe() -> None:
    secret = "local path with private details"

    def fail(_path: Path) -> tuple[Any, int]:
        raise RuntimeError(secret)

    runner = PyannoteVad(
        model_name="pyannote/segmentation-3.0",
        device="cpu",
        pipeline_provider=lambda: object(),
        waveform_loader=fail,
    )

    with pytest.raises(VadInputError) as error:
        runner.run(Path("not-opened.wav"))

    assert str(error.value) == "Unable to read the normalized WAV."
    assert secret not in str(error.value)
