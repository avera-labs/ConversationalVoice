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
    def __init__(self, *, data=None, content=b"", status_code=200):
        self.data = data
        self.content = content
        self.status_code = status_code

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


def test_audio_tags_uses_configured_gemini_model_and_wav(policy):
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
    assert payload["model"] == "google/gemini-3.7-flash"
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
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["max_tokens"] == 2048
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


@pytest.mark.parametrize(
    ("source_text", "provider_text", "repaired_text"),
    [
        (
            "我想拍你我说好导演就坐在那边想了很久，",
            "[calm]我想拍你，我说好，导演就坐在那边想了很久，",
            "[calm]我想拍你我说好导演就坐在那边想了很久，",
        ),
        ("你好 世界", "你好， [calm]世界", "你好 [calm]世界"),
    ],
)
def test_audio_annotation_auto_repairs_punctuation_and_whitespace(
    policy, caplog, source_text, provider_text, repaired_text
):
    transport = Transport(
        Response(
            data={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "text_with_audio_tags": provider_text,
                                    "instruction": "Speak calmly.",
                                }
                            )
                        }
                    }
                ],
                "usage": {},
            }
        )
    )

    annotation, _usage = AudioTagsClient(
        policy.audio_tags, "key", transport=transport
    ).analyze(b"wav", source_text)

    assert len(transport.calls) == 1
    assert annotation["text"] == source_text
    assert annotation["text_with_audio_tags"] == repaired_text
    repair_log = next(
        record
        for record in caplog.records
        if record.getMessage().startswith(
            "openrouter_audio_tags.derived_text_auto_repaired"
        )
    )
    assert repair_log.levelname == "WARNING"
    assert f"expected_text={source_text!r}" in repair_log.getMessage()
    assert f"original_text_with_audio_tags={provider_text!r}" in repair_log.getMessage()
    assert f"repaired_text_with_audio_tags={repaired_text!r}" in repair_log.getMessage()


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


def test_audio_tags_logs_openrouter_400_message(policy, caplog):
    transport = Transport(
        Response(
            status_code=400,
            data={
                "error": {
                    "message": "Invalid reasoning effort",
                    "metadata": {
                        "raw": json.dumps(
                            {
                                "error": {
                                    "message": "effort none is not supported"
                                }
                            }
                        )
                    },
                }
            },
        )
    )

    with pytest.raises(
        OpenRouterProviderError,
        match="^openrouter_audio_tags_http_400$",
    ):
        AudioTagsClient(policy.audio_tags, "key", transport=transport).analyze(
            b"wav", "hello"
        )

    error_log = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("openrouter_audio_tags.http_error")
    )
    assert error_log.levelname == "ERROR"
    assert "status=400" in error_log.getMessage()
    assert (
        "provider_message='Invalid reasoning effort | effort none is not supported'"
        in error_log.getMessage()
    )


def test_audio_annotation_retries_with_derived_text_failure_reason(policy, caplog):
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
    assert len(transport.calls) == 2
    retry_text = transport.calls[1][1]["json"]["messages"][1]["content"][1]["text"]
    assert "CORRECTION_REQUIRED" in retry_text
    assert "reason_code: derived_text_mismatch" in retry_text
    mismatch = next(
        record
        for record in caplog.records
        if record.getMessage().startswith(
            "openrouter_audio_tags.derived_text_mismatch"
        )
    )
    assert mismatch.levelname == "WARNING"
    assert "attempt=1/3" in mismatch.getMessage()
    assert "difference_index=0" in mismatch.getMessage()
    assert "expected_character='h'(U+0068)" in mismatch.getMessage()
    assert "actual_character='c'(U+0063)" in mismatch.getMessage()
    assert "expected_text='hello'" in mismatch.getMessage()
    assert "actual_derived_text='changed'" in mismatch.getMessage()
    assert "text_with_audio_tags='[sighs]changed'" in mismatch.getMessage()


def test_final_derived_text_mismatch_is_logged_as_error(policy, caplog):
    invalid = Response(
        data={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "text_with_audio_tags": "hallo[calm]",
                                "instruction": "Speak calmly.",
                            }
                        )
                    }
                }
            ],
            "usage": {},
        }
    )
    one_attempt = policy.audio_tags.model_copy(update={"max_attempts": 1})

    with pytest.raises(
        OpenRouterProviderError,
        match="^openrouter_audio_tags_derived_text_mismatch$",
    ):
        AudioTagsClient(one_attempt, "key", transport=Transport(invalid)).analyze(
            b"wav", "hello"
        )

    mismatch = next(
        record
        for record in caplog.records
        if record.getMessage().startswith(
            "openrouter_audio_tags.derived_text_mismatch"
        )
    )
    assert mismatch.levelname == "ERROR"
    assert "attempt=1/1" in mismatch.getMessage()
    assert "difference_index=1" in mismatch.getMessage()
    assert "expected_character='e'(U+0065)" in mismatch.getMessage()
    assert "actual_character='a'(U+0061)" in mismatch.getMessage()


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
