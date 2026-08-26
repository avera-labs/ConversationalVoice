import io
import wave
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from voice_pipeline_forced_alignment import Qwen3SegmentAligner


def wav(duration_ms=1000):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\1\0" * round(duration_ms * 16))
    return target.getvalue()


@dataclass
class Item:
    text: str
    start_time: float
    end_time: float


class Model:
    timestamp_segment_time = 80

    def __init__(self):
        self.calls = []

    def align(self, **kwargs):
        self.calls.append(kwargs)
        return [[Item("Hello", 0.1, 0.4), Item("world", 0.5, 0.9)]]


def test_aligner_uses_plain_text_and_merges_inline_tags():
    model = Model()
    aligner = Qwen3SegmentAligner(
        SimpleNamespace(), model_factory=lambda _policy: model
    )

    result = aligner.align(
        wav(),
        text_with_audio_tags="[calm]Hello [sighs]world.",
        language="en",
    )

    assert model.calls[0]["text"] == "Hello world."
    assert model.calls[0]["language"] == "English"
    assert [item["type"] for item in result] == [
        "audio_tag",
        "word",
        "audio_tag",
        "word",
    ]


@pytest.mark.parametrize(
    ("language", "provider_language"),
    [
        ("yue", "Cantonese"),
        ("de", "German"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("pt", "Portuguese"),
        ("ru", "Russian"),
    ],
)
def test_passes_each_model_supported_language(language, provider_language):
    model = Model()
    aligner = Qwen3SegmentAligner(
        SimpleNamespace(), model_factory=lambda _policy: model
    )
    aligner.align(wav(), text_with_audio_tags="Hello world", language=language)
    assert model.calls[0]["language"] == provider_language


def test_rejects_language_only_at_forced_alignment_boundary():
    aligner = Qwen3SegmentAligner(
        SimpleNamespace(), model_factory=lambda _policy: Model()
    )
    with pytest.raises(ValueError, match="forced-alignment language is unsupported"):
        aligner.align(wav(), text_with_audio_tags="Hallo", language="nl")


def test_tag_only_segment_does_not_load_model():
    aligner = Qwen3SegmentAligner(
        SimpleNamespace(),
        model_factory=lambda _policy: (_ for _ in ()).throw(AssertionError()),
    )

    assert aligner.align(wav(), text_with_audio_tags="[sighs]", language="zh") == [
        {
            "item_index": 0,
            "type": "audio_tag",
            "text": "[sighs]",
            "text_start": 0,
            "text_end": 0,
            "start_ms": 0,
            "end_ms": 0,
        }
    ]


def test_timestamp_within_one_model_step_is_clamped_to_wav_duration():
    class QuantizedModel(Model):
        def align(self, **kwargs):
            self.calls.append(kwargs)
            return [[Item("Hello", 0.1, 1.04)]]

    model = QuantizedModel()
    aligner = Qwen3SegmentAligner(
        SimpleNamespace(), model_factory=lambda _policy: model
    )

    result = aligner.align(wav(), text_with_audio_tags="Hello", language="en")

    assert result[0]["start_ms"] == 100
    assert result[0]["end_ms"] == 1000


def test_timestamp_beyond_one_model_step_is_rejected_with_diagnostics():
    class InvalidModel(Model):
        def align(self, **kwargs):
            self.calls.append(kwargs)
            return [[Item("Hello", 0.1, 1.081)]]

    model = InvalidModel()
    aligner = Qwen3SegmentAligner(
        SimpleNamespace(), model_factory=lambda _policy: model
    )

    with pytest.raises(
        RuntimeError,
        match=("start_ms=100, end_ms=1081, duration_ms=1000, tolerance_ms=80"),
    ):
        aligner.align(wav(), text_with_audio_tags="Hello", language="en")
