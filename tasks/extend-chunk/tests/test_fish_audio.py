import base64
import io
import wave

import pytest

from voice_pipeline_extend_chunk.fish_audio import (
    OpenRouterFishAudioClient,
    OpenRouterFishAudioError,
    tts_text,
)


class Response:
    def __init__(self, *, content=b"", data=None, status_code=200):
        self.content = content
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


class Transport:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/transcriptions"):
            return Response(data={"text": " Reference  words. "})
        return Response(content=b"\1\0" * 44100)


def test_reference_asr_and_voice_clone_use_openrouter_json_apis(policy):
    transport = Transport()
    client = OpenRouterFishAudioClient(
        policy.fish_audio, "openrouter-key", transport=transport
    )
    assert client.transcribe_reference(b"wav") == "Reference words."
    output = client.synthesize("[laughs] Hi", b"wav", "Reference words.")

    _, asr_request = transport.calls[0]
    assert asr_request["headers"] == {
        "Authorization": "Bearer openrouter-key",
        "Content-Type": "application/json",
    }
    assert asr_request["json"] == {
        "model": "fish-audio/transcribe-1",
        "input_audio": {
            "data": base64.b64encode(b"wav").decode("ascii"),
            "format": "wav",
        },
        "language": "en",
    }

    _, tts_request = transport.calls[1]
    assert tts_request["headers"] == {
        "Authorization": "Bearer openrouter-key",
        "Content-Type": "application/json",
    }
    payload = tts_request["json"]
    assert payload["model"] == "fish-audio/s2.1-pro"
    assert payload["input"] == "[laughs] Hi"
    assert payload["input_references"] == [
        {
            "type": "input_audio",
            "input_audio": {
                "data": "data:audio/wav;base64,"
                + base64.b64encode(b"wav").decode("ascii")
            },
        },
        {"type": "text", "text": "Reference words."},
    ]
    assert payload["response_format"] == "pcm"
    assert payload["provider"]["options"]["fish-audio"]["sample_rate"] == 44100
    with wave.open(io.BytesIO(output), "rb") as reader:
        assert reader.getframerate() == 44100
        assert reader.getnframes() == 44100


def test_tts_text_keeps_portable_square_bracket_tags_separate_from_words():
    assert (
        tts_text({"audio_tags": ["[laughs]", "[excited]"], "text": "We did it!"})
        == "[laughs] [excited] We did it!"
    )


def test_deterministic_fish_error_is_not_retried(policy):
    class FailedTransport:
        calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return Response(status_code=422)

    transport = FailedTransport()
    client = OpenRouterFishAudioClient(policy.fish_audio, "key", transport=transport)

    with pytest.raises(
        OpenRouterFishAudioError, match="openrouter_fish_audio_http_422"
    ):
        client.synthesize("Hello", b"wav", "Reference words.")
    assert transport.calls == 1
