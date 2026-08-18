import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import voice_pipeline_diarize_audio_part.diarization as diarization_module
from voice_pipeline_diarize_audio_part.diarization import (
    DiarizationEngine,
    apply_torch_26_checkpoint_compatibility,
    diarizen_initialization_device,
    extract_turns,
    resolve_device,
)


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.emptied = False

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, _index: int) -> str:
        return "Test GPU"

    def empty_cache(self) -> None:
        self.emptied = True


class FakeTorch:
    def __init__(self, cuda: bool) -> None:
        self.cuda = FakeCuda(cuda)


class Annotation:
    def itertracks(self, *, yield_label: bool):
        assert yield_label
        yield SimpleNamespace(start=0.1, end=0.9), None, "speaker-a"


def test_extracts_direct_and_compatibility_annotations() -> None:
    assert extract_turns(Annotation())[0].speaker_label == "speaker-a"
    assert (
        extract_turns(SimpleNamespace(speaker_diarization=Annotation()))[0].start == 0.1
    )


def test_rejects_unknown_output() -> None:
    with pytest.raises(TypeError):
        extract_turns(object())


def test_checkpoint_compatibility_only_allowlists_torch_version() -> None:
    observed: list[list[object]] = []
    torch_module = SimpleNamespace(
        serialization=SimpleNamespace(add_safe_globals=observed.append),
        torch_version=SimpleNamespace(TorchVersion=object),
    )
    apply_torch_26_checkpoint_compatibility(
        torch_module, pyannote_types=(str, int, float)
    )
    assert observed == [[object, str, int, float]]


def test_auto_device_prefers_cuda_and_allows_cpu() -> None:
    assert resolve_device("auto", FakeTorch(True)).device == "cuda"
    assert resolve_device("auto", FakeTorch(False)).device == "cpu"


def test_cpu_device_is_selected_during_initialization_without_moving() -> None:
    torch = FakeTorch(True)
    assert torch.cuda.is_available() is True
    with diarizen_initialization_device(torch, "cpu"):
        assert torch.cuda.is_available() is False
    assert torch.cuda.is_available() is True


def test_official_factory_initializes_on_cpu_without_moving(monkeypatch) -> None:
    observed: list[tuple[str, bool]] = []
    torch = FakeTorch(True)

    class Pipeline:
        def to(self, _device):
            raise AssertionError("the initialized pipeline must not be moved")

    class PipelineType:
        @staticmethod
        def from_pretrained(model: str):
            observed.append((model, torch.cuda.is_available()))
            return Pipeline()

    diarizen = ModuleType("diarizen")
    diarizen.__path__ = []
    pipelines = ModuleType("diarizen.pipelines")
    pipelines.__path__ = []
    inference = ModuleType("diarizen.pipelines.inference")
    inference.DiariZenPipeline = PipelineType
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diarizen", diarizen)
    monkeypatch.setitem(sys.modules, "diarizen.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "diarizen.pipelines.inference", inference)
    monkeypatch.setattr(
        diarization_module, "apply_torch_26_checkpoint_compatibility", lambda _: None
    )

    pipeline = diarization_module._load_pipeline("owner/model", "cpu")

    assert isinstance(pipeline, Pipeline)
    assert observed == [("owner/model", False)]
    assert torch.cuda.is_available() is True


def test_pipeline_is_loaded_lazily_and_cached(capsys) -> None:
    loads: list[tuple[str, str]] = []

    class Pipeline:
        def __call__(self, path: str):
            print("hidden local cache path")
            assert path.endswith("audio.wav")
            return Annotation()

    def loader(model: str, device: str):
        loads.append((model, device))
        return Pipeline()

    torch = FakeTorch(False)
    engine = DiarizationEngine(
        model="owner/model",
        requested_device="auto",
        loader=loader,
        torch_module=torch,
    )
    first = engine.infer(Path("audio.wav"))
    second = engine.infer(Path("audio.wav"))
    assert first.model_cache_hit is False
    assert second.model_cache_hit is True
    assert len(loads) == 1
    assert "hidden local cache path" not in capsys.readouterr().out
    engine.close()
