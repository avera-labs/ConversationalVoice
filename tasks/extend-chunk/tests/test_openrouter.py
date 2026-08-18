import json

import pytest

from voice_pipeline_extend_chunk.openrouter import OpenRouterClient, OpenRouterError


class Response:
    status_code = 200

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "utterances": [
                                    {
                                        "utterance_index": 0,
                                        "speaker_id": 0,
                                        "text": "A continuation.",
                                        "tone": "calm",
                                        "type": "dialogue",
                                        "placement": "sequential",
                                        "audio_tags": [],
                                    }
                                ]
                            }
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.01,
            },
        }


class Transport:
    def post(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs
        return Response()


class ErrorResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class SequenceTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *_args, **_kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_request_uses_strict_schema_and_two_minute_guidance(policy):
    transport = Transport()
    client = OpenRouterClient(policy.openrouter, "key", transport=transport)
    wire, usage = client.extend(
        {
            "scene": {},
            "speakers": [],
            "speaker_mapping": [
                {"output_slot": 0, "diarization_speaker_id": 4},
                {"output_slot": 1, "diarization_speaker_id": 7},
            ],
        },
        {"speakers": []},
        policy.dialogue,
    )
    payload = transport.kwargs["json"]
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["plugins"] == [{"id": "response-healing"}]
    assert payload["stream"] is False
    assert (
        payload["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
    )
    schema = payload["response_format"]["json_schema"]["schema"]
    speaker_id_schema = schema["properties"]["utterances"]["items"]["properties"][
        "speaker_id"
    ]
    assert speaker_id_schema == {"type": "integer", "minimum": 0, "maximum": 1}
    assert schema["properties"]["utterances"]["minItems"] == 8
    assert "maxItems" not in schema["properties"]["utterances"]
    encoded_schema = json.dumps(schema)
    assert "uniqueItems" not in encoded_schema
    assert "minLength" not in encoded_schema
    assert "approximately 120 seconds or 300" in payload["messages"][0]["content"]
    assert "[laughs]" in payload["messages"][0]["content"]
    assert '"persona_speaker_id":"4"' in payload["messages"][1]["content"]
    assert wire["utterances"][0]["speaker_id"] == 0
    assert usage["total_tokens"] == 30


def test_retryable_status_uses_bounded_retry(policy):
    transport = SequenceTransport([ErrorResponse(429), Response()])
    sleeps = []
    client = OpenRouterClient(
        policy.openrouter, "key", transport=transport, sleeper=sleeps.append
    )

    wire, _usage = client.extend(
        {
            "speaker_mapping": [
                {"output_slot": 0, "diarization_speaker_id": 4},
                {"output_slot": 1, "diarization_speaker_id": 7},
            ]
        },
        {},
        policy.dialogue,
    )

    assert wire["utterances"][0]["speaker_id"] == 0
    assert transport.calls == 2
    assert sleeps == [policy.openrouter.retry_backoff_seconds]


def test_deterministic_provider_error_is_not_retried(policy):
    transport = SequenceTransport([ErrorResponse(400)])
    client = OpenRouterClient(policy.openrouter, "key", transport=transport)

    with pytest.raises(OpenRouterError, match="openrouter_http_400"):
        client.extend(
            {
                "speaker_mapping": [
                    {"output_slot": 0, "diarization_speaker_id": 4},
                    {"output_slot": 1, "diarization_speaker_id": 7},
                ]
            },
            {},
            policy.dialogue,
        )
    assert transport.calls == 1


def test_structured_content_object_is_accepted(policy):
    class ObjectResponse(Response):
        def json(self):
            data = super().json()
            data["choices"][0]["message"]["content"] = {
                "utterances": [{"speaker_id": 0}]
            }
            return data

    transport = SequenceTransport([ObjectResponse()])
    client = OpenRouterClient(policy.openrouter, "key", transport=transport)

    wire, _usage = client.extend(
        {
            "speaker_mapping": [
                {"output_slot": 0, "diarization_speaker_id": 4},
                {"output_slot": 1, "diarization_speaker_id": 7},
            ]
        },
        {},
        policy.dialogue,
    )

    assert wire == {"utterances": [{"speaker_id": 0}]}


def test_invalid_structured_content_reports_precise_safe_error(policy):
    class InvalidResponse(Response):
        def json(self):
            data = super().json()
            data["choices"][0]["message"]["content"] = "not-json"
            return data

    transport = SequenceTransport(
        [InvalidResponse() for _attempt in range(policy.openrouter.max_attempts)]
    )
    client = OpenRouterClient(policy.openrouter, "key", transport=transport)

    with pytest.raises(OpenRouterError, match="openrouter_structured_content_not_json"):
        client.extend(
            {
                "speaker_mapping": [
                    {"output_slot": 0, "diarization_speaker_id": 4},
                    {"output_slot": 1, "diarization_speaker_id": 7},
                ]
            },
            {},
            policy.dialogue,
        )


def test_invalid_schema_400_has_safe_specific_error(policy):
    class InvalidSchemaResponse(ErrorResponse):
        def json(self):
            return {"error": {"message": "Invalid response schema keyword."}}

    client = OpenRouterClient(
        policy.openrouter,
        "key",
        transport=SequenceTransport([InvalidSchemaResponse(400)]),
    )

    with pytest.raises(OpenRouterError, match="openrouter_http_400_invalid_schema"):
        client.extend(
            {
                "speaker_mapping": [
                    {"output_slot": 0, "diarization_speaker_id": 4},
                    {"output_slot": 1, "diarization_speaker_id": 7},
                ]
            },
            {},
            policy.dialogue,
        )


def test_nested_provider_schema_error_has_safe_specific_error(policy):
    class InvalidSchemaResponse(ErrorResponse):
        def json(self):
            return {
                "error": {
                    "message": "Provider returned error",
                    "metadata": {
                        "raw": json.dumps(
                            {
                                "error": {
                                    "message": (
                                        "schema at properties.utterances.items "
                                        "requires unspecified property"
                                    )
                                }
                            }
                        )
                    },
                }
            }

    client = OpenRouterClient(
        policy.openrouter,
        "key",
        transport=SequenceTransport([InvalidSchemaResponse(400)]),
    )

    with pytest.raises(OpenRouterError, match="openrouter_http_400_invalid_schema"):
        client.extend(
            {
                "speaker_mapping": [
                    {"output_slot": 0, "diarization_speaker_id": 4},
                    {"output_slot": 1, "diarization_speaker_id": 7},
                ]
            },
            {},
            policy.dialogue,
        )
