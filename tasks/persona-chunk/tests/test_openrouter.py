import json

import pytest

from voice_pipeline_persona_chunk.openrouter import (
    ENDPOINT,
    OpenRouterClient,
    OpenRouterError,
)
from voice_pipeline_persona_chunk.prompt import build_response_schema


def speaker():
    return {
        "name": None,
        "age": None,
        "ethnicity": None,
        "gender": None,
        "tag": "You enjoy having a good conversation.",
        "alpha": "low",
        "evidence": None,
        "primary_emotion": "neutral",
        "secondary_emotion": None,
        "emotion_intensity": "low",
        "laugh": False,
        "cry": False,
        "whisper": False,
        "shout": False,
        "sigh": False,
        "overall_tone": "calm",
    }


class Response:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.body = body or {}

    def json(self):
        return self.body


class Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


def success():
    arguments = {
        "scene": {
            "description": "A calm exchange.",
            "overall_tone": "calm",
            "emotion_intensity": "low",
        },
        "speakers": {"4": speaker(), "7": speaker()},
    }
    return Response(
        body={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(arguments),
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.001,
            },
        }
    )


def test_request_uses_structured_output_and_schema_in_system_prompt(policy):
    transport = Transport([success()])
    client = OpenRouterClient(
        policy.openrouter, "secret", transport, sleeper=lambda _: None
    )
    wire, usage = client.analyze(b"mp3", "SRT", (4, 7))
    args, kwargs = transport.calls[0]
    assert args == (ENDPOINT,)
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    payload = kwargs["json"]
    assert payload["model"] == "xiaomi/mimo-v2.5"
    schema = build_response_schema((4, 7))
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "persona_analysis",
            "strict": True,
            "schema": schema,
        },
    }
    assert payload["messages"][0]["role"] == "system"
    assert (
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        in payload["messages"][0]["content"]
    )
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert payload["provider"] == {"require_parameters": True, "allow_fallbacks": True}
    assert set(wire["speakers"]) == {"4", "7"}
    assert usage["total_tokens"] == 15


def test_transient_error_retries_but_deterministic_4xx_does_not(policy):
    transport = Transport([Response(429), success()])
    client = OpenRouterClient(
        policy.openrouter, "secret", transport, sleeper=lambda _: None
    )
    client.analyze(b"mp3", "SRT", (4, 7))
    assert len(transport.calls) == 2

    transport = Transport([Response(400), success()])
    client = OpenRouterClient(
        policy.openrouter, "secret", transport, sleeper=lambda _: None
    )
    with pytest.raises(OpenRouterError, match="openrouter_http_400"):
        client.analyze(b"mp3", "SRT", (4, 7))
    assert len(transport.calls) == 1


def test_structured_output_is_not_revalidated_by_client(policy):
    invalid = success()
    arguments = json.loads(invalid.body["choices"][0]["message"]["content"])
    arguments["speakers"].pop("7")
    invalid.body["choices"][0]["message"]["content"] = json.dumps(arguments)
    transport = Transport([invalid])
    client = OpenRouterClient(
        policy.openrouter, "secret", transport, sleeper=lambda _: None
    )
    wire, _usage = client.analyze(b"mp3", "SRT", (4, 7))
    assert set(wire["speakers"]) == {"4"}
    assert len(transport.calls) == 1
