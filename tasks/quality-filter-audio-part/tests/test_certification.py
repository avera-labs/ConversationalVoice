from types import SimpleNamespace

import pytest

from voice_pipeline_quality_filter_audio_part import certification


@pytest.mark.parametrize(
    ("reported", "normalized"),
    [("x86_64", "x86_64"), ("AMD64", "x86_64"), ("aarch64", "aarch64"), ("arm64", "aarch64")],
)
def test_supported_architecture_and_cpu_are_detected(monkeypatch, reported, normalized) -> None:
    monkeypatch.setattr(certification.platform, "machine", lambda: reported)
    monkeypatch.setattr(certification.platform, "processor", lambda: "test-cpu")
    monkeypatch.setitem(
        __import__("sys").modules,
        "tensorflow",
        SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda _kind: [])),
    )
    assert certification.detect_execution_target() == certification.ExecutionTarget(
        normalized, "cpu", "test-cpu"
    )


def test_gpu_is_selected_automatically(monkeypatch) -> None:
    monkeypatch.setattr(certification.platform, "machine", lambda: "x86_64")
    device = SimpleNamespace(name="/physical_device:GPU:0")
    monkeypatch.setitem(
        __import__("sys").modules,
        "tensorflow",
        SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda _kind: [device])),
    )
    assert certification.detect_execution_target().device == "gpu"


def test_unsupported_architecture_fails_before_tensorflow(monkeypatch) -> None:
    monkeypatch.setattr(certification.platform, "machine", lambda: "riscv64")
    with pytest.raises(RuntimeError, match="architecture"):
        certification.detect_execution_target()


def test_tensorflow_detection_failure_is_not_silently_downgraded(monkeypatch) -> None:
    monkeypatch.setattr(certification.platform, "machine", lambda: "x86_64")

    class BrokenConfig:
        @staticmethod
        def list_physical_devices(_kind):
            raise RuntimeError("driver details")

    monkeypatch.setitem(
        __import__("sys").modules,
        "tensorflow",
        SimpleNamespace(config=BrokenConfig()),
    )
    with pytest.raises(RuntimeError, match="discovery failed"):
        certification.detect_execution_target()
