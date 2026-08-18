from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from voice_pipeline_transcribe_chunk.model import ParakeetModel


class FakeModel:
    def __init__(self, result):
        self.result = result

    def transcribe(self, _audio, timestamps):
        assert timestamps is True
        return ([self.result], [self.result])


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


def adapter(result):
    model = ParakeetModel(SimpleNamespace())
    model._model = FakeModel(result)
    model._torch = FakeTorch()
    return model


def test_ne_mo_tuple_output_is_normalized():
    result = SimpleNamespace(
        timestep={"word": [{"word": "Hello", "start": 0.1, "end": 0.4}]},
        word_confidence=[0.9],
    )
    words = adapter(result).transcribe(np.zeros(16000, dtype=np.float32))
    assert words[0].text == "Hello"
    assert words[0].confidence == 0.9


def test_confidence_count_mismatch_is_a_failure():
    result = SimpleNamespace(
        timestep={"word": [{"word": "Hello", "start": 0.1, "end": 0.4}]},
        word_confidence=[],
    )
    with pytest.raises(RuntimeError, match="confidence count mismatch"):
        adapter(result).transcribe(np.zeros(16000, dtype=np.float32))
