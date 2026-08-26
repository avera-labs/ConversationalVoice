import base64
import io
import wave

import pytest

from voice_pipeline_extend_chunk.fish_audio import (
    OpenRouterFishAudioClient,
    OpenRouterFishAudioError,
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
    output = client.synthesize(
        {
            "text": "Hi",
            "text_with_audio_tags": "[laughs] Hi",
            "instruction": "Laugh, then greet the listener.",
        },
        b"wav",
        "Reference words.",
    )

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


def test_reference_asr_uses_chinese_language(policy):
    transport = Transport()
    client = OpenRouterFishAudioClient(
        policy.fish_audio, "openrouter-key", transport=transport
    )

    client.transcribe_reference(b"wav", language="zh")

    assert transport.calls[0][1]["json"]["language"] == "zh"


def test_reference_asr_forwards_language_without_pipeline_filter(policy):
    transport = Transport()
    client = OpenRouterFishAudioClient(
        policy.fish_audio, "openrouter-key", transport=transport
    )

    client.transcribe_reference(b"wav", language="x-unsupported")

    assert transport.calls[0][1]["json"]["language"] == "x-unsupported"


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
        client.synthesize(
            {
                "text": "Hello",
                "text_with_audio_tags": "Hello",
                "instruction": "Speak naturally.",
            },
            b"wav",
            "Reference words.",
        )
    assert transport.calls == 1


def test_unmapped_tts_model_uses_plain_text_without_fish_options(policy):
    transport = Transport()
    tts_policy = policy.fish_audio.model_copy(update={"model": "provider/plain-tts"})
    client = OpenRouterFishAudioClient(tts_policy, "key", transport=transport)

    client.synthesize(
        {
            "text": "Hello",
            "text_with_audio_tags": "[laughs]Hello",
            "instruction": "Laugh while greeting the listener.",
        },
        b"wav",
        "Reference words.",
    )

    payload = transport.calls[0][1]["json"]
    assert payload["model"] == "provider/plain-tts"
    assert payload["input"] == "Hello"
    assert "provider" not in payload


def test_temporarily_unmapped_mimo_uses_plain_text(policy, caplog):
    transport = Transport()
    tts_policy = policy.fish_audio.model_copy(
        update={"model": "mimo-v2.5-tts-voiceclone"}
    )
    client = OpenRouterFishAudioClient(tts_policy, "key", transport=transport)

    client.synthesize(
        {
            "text": "Hello",
            "text_with_audio_tags": "[laughs]Hello",
            "instruction": "Laugh while greeting the listener.",
        },
        b"wav",
        "Reference words.",
    )

    payload = transport.calls[0][1]["json"]
    assert payload["input"] == "Hello"
    assert "provider" not in payload
    assert "missing from TTS_MODEL_CAPABILITIES" in caplog.text
