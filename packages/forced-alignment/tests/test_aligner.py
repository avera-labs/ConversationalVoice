import io
import wave
from dataclasses import dataclass
from types import SimpleNamespace

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


def test_tag_only_segment_does_not_load_model():
    aligner = Qwen3SegmentAligner(
        SimpleNamespace(),
        model_factory=lambda _policy: (_ for _ in ()).throw(AssertionError()),
    )

    assert aligner.align(
        wav(), text_with_audio_tags="[sighs]", language="zh"
    ) == [
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
