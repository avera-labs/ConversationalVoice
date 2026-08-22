import base64
import json

import pytest

from voice_pipeline_reconstruct_chunk.providers import (
    AudioTagsClient,
    FishAudioClient,
    OpenRouterProviderError,
)


class Response:
    status_code = 200

    def __init__(self, *, data=None, content=b""):
        self.data = data
        self.content = content

    def json(self):
        return self.data


class Transport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_audio_tags_uses_configured_mimo_model_and_wav(policy):
    transport = Transport(
        Response(
            data={
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"audio_tags":["[sighs]"],"tone":"tired"}\n```'
                        }
                    }
                ],
                "usage": {},
            }
        )
    )
    tags, _usage = AudioTagsClient(
        policy.audio_tags, "key", transport=transport
    ).analyze(b"wav", "hello")
    payload = transport.calls[0][1]["json"]
    assert payload["model"] == "xiaomi/mimo-v2.5"
    schema = payload["response_format"]["json_schema"]["schema"]
    schema_json = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["messages"][0]["content"].endswith(schema_json)
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert (
        base64.b64decode(payload["messages"][1]["content"][0]["input_audio"]["data"])
        == b"wav"
    )
    assert tags == {"audio_tags": ["[sighs]"], "tone": "tired"}


def test_audio_tags_reports_safe_empty_content_shape(policy):
    transport = Transport(
        Response(
            data={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": None, "reasoning": "private output"},
                    }
                ]
            }
        )
    )

    with pytest.raises(
        OpenRouterProviderError,
        match=(
            "^openrouter_audio_tags_invalid_response_content_null_"
            "finish_length_reasoning_present$"
        ),
    ):
        AudioTagsClient(policy.audio_tags, "key", transport=transport).analyze(
            b"wav", "secret transcript"
        )


def test_tts_uses_fish_audio_with_audio_reference_and_no_asr(policy):
    transport = Transport(Response(content=b"\x01\x00" * 441))
    result = FishAudioClient(policy.tts, "key", transport=transport).synthesize(
        "hello", b"reference"
    )
    url, request = transport.calls[0]
    payload = request["json"]
    assert url.endswith("/audio/speech")
    assert payload["model"] == "fish-audio/s2.1-pro"
    assert payload["input_references"] == [
        {
            "type": "input_audio",
            "input_audio": {
                "data": "data:audio/wav;base64,"
                + base64.b64encode(b"reference").decode()
            },
        }
    ]
    assert result.startswith(b"RIFF")
