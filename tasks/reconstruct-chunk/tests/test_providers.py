import base64
import json

import pytest
from voice_pipeline_chunk_contracts import AUDIO_TAGS

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


class SequenceTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_audio_tags_uses_configured_mimo_model_and_wav(policy):
    transport = Transport(
        Response(
            data={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '```json\n{"text_with_audio_tags":"[sighs]hello",'
                                '"instruction":"Speak with audible fatigue."}\n```'
                            )
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
    assert set(schema["properties"]) == {
        "text_with_audio_tags",
        "instruction",
    }
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["messages"][0]["content"].endswith(schema_json)
    assert (
        json.dumps(sorted(AUDIO_TAGS), ensure_ascii=False, separators=(",", ":"))
        in payload["messages"][0]["content"]
    )
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert (
        base64.b64decode(payload["messages"][1]["content"][0]["input_audio"]["data"])
        == b"wav"
    )
    assert tags == {
        "text": "hello",
        "text_with_audio_tags": "[sighs]hello",
        "instruction": "Speak with audible fatigue.",
    }


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


def test_audio_annotation_retries_with_derived_text_failure_reason(policy):
    invalid = Response(
        data={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "text_with_audio_tags": "[sighs]changed",
                                "instruction": "Speak tiredly.",
                            }
                        )
                    }
                }
            ],
            "usage": {},
        }
    )
    valid = Response(
        data={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "text_with_audio_tags": "[sighs]hello",
                                "instruction": "Speak tiredly.",
                            }
                        )
                    }
                }
            ],
            "usage": {},
        }
    )
    transport = SequenceTransport([invalid, valid])

    annotation, _usage = AudioTagsClient(
        policy.audio_tags, "key", transport=transport
    ).analyze(b"wav", "hello")

    assert annotation["text_with_audio_tags"] == "[sighs]hello"
    retry_text = transport.calls[1][1]["json"]["messages"][1]["content"][1]["text"]
    assert "CORRECTION_REQUIRED" in retry_text
    assert "reason_code: derived_text_mismatch" in retry_text


def test_tts_uses_fish_audio_with_audio_reference_and_no_asr(policy):
    transport = Transport(Response(content=b"\x01\x00" * 441))
    result = FishAudioClient(policy.tts, "key", transport=transport).synthesize(
        {
            "text": "hello",
            "text_with_audio_tags": "[calm]hello",
            "instruction": "Speak calmly.",
        },
        b"reference",
    )
    url, request = transport.calls[0]
    payload = request["json"]
    assert url.endswith("/audio/speech")
    assert payload["model"] == "fish-audio/s2.1-pro"
    assert payload["input"] == "[calm]hello"
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


def test_unmapped_tts_model_uses_plain_text_without_fish_options(policy):
    transport = Transport(Response(content=b"\x01\x00" * 441))
    tts_policy = policy.tts.model_copy(update={"model": "provider/plain-tts"})

    FishAudioClient(tts_policy, "key", transport=transport).synthesize(
        {
            "text": "hello",
            "text_with_audio_tags": "[calm]hello",
            "instruction": "Speak calmly.",
        },
        b"reference",
    )

    payload = transport.calls[0][1]["json"]
    assert payload["model"] == "provider/plain-tts"
    assert payload["input"] == "hello"
    assert "provider" not in payload


def test_temporarily_unmapped_mimo_uses_plain_text(policy, caplog):
    transport = Transport(Response(content=b"\x01\x00" * 441))
    tts_policy = policy.tts.model_copy(update={"model": "mimo-v2.5-tts-voiceclone"})

    FishAudioClient(tts_policy, "key", transport=transport).synthesize(
        {
            "text": "hello",
            "text_with_audio_tags": "[calm]hello",
            "instruction": "Speak calmly.",
        },
        b"reference",
    )

    payload = transport.calls[0][1]["json"]
    assert payload["input"] == "hello"
    assert "provider" not in payload
    assert "missing from TTS_MODEL_CAPABILITIES" in caplog.text
