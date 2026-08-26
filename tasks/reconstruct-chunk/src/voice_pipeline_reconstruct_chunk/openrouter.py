from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
from voice_pipeline_chunk_contracts import AUDIO_TAGS

from .errors import OpenRouterProviderError

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class AudioTagsClient:
    def __init__(self, policy, api_key: str, transport=None, sleeper=time.sleep):
        self.policy = policy
        self.api_key = api_key
        self.transport = transport or httpx.Client()
        self._owns_transport = transport is None
        self.sleeper = sleeper

    def analyze(self, audio: bytes, text: str) -> tuple[dict, dict]:
        if not audio or not text:
            raise ValueError("audio tag input is incomplete")
        payload = self._payload(audio, text)
        data = self._request(payload)
        try:
            choices = data["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("invalid choice count")
            wire = _parse_json_content(choices[0]["message"]["content"])
            if not isinstance(wire, dict) or set(wire) != {"audio_tags", "tone"}:
                raise ValueError("invalid audio tag fields")
            tags = wire["audio_tags"]
            tone = wire["tone"]
            if (
                not isinstance(tags, list)
                or len(tags) > 3
                or len(set(tags)) != len(tags)
                or any(tag not in AUDIO_TAGS for tag in tags)
                or not isinstance(tone, str)
                or tone != tone.strip()
            ):
                raise ValueError("invalid audio tag response")
            usage = _usage(data.get("usage"), self.policy.model)
            return {"audio_tags": tags, "tone": tone}, usage
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OpenRouterProviderError(
                "openrouter_audio_tags_invalid_response_" + _response_shape(data)
            ) from exc

    def _payload(self, audio: bytes, text: str) -> dict:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "audio_tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(AUDIO_TAGS)},
                    "maxItems": 3,
                },
                "tone": {"type": "string"},
            },
            "required": ["audio_tags", "tone"],
        }
        schema_json = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return {
            "model": self.policy.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Analyze only the supplied utterance audio. Return zero to three "
                        "approved tags describing audible emotion or paralinguistic events. "
                        "Do not infer events that are not audible. tone is a concise delivery "
                        "description and may be empty. Return only one JSON object with no "
                        "Markdown or additional commentary. The response must follow the "
                        "JSON Schema appended below.\n\nJSON_SCHEMA:\n" + schema_json
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio).decode("ascii"),
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": f"UTTERANCE TEXT:\n{text}"},
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "source_utterance_audio_tags",
                    "strict": True,
                    "schema": schema,
                },
            },
            "reasoning": {"effort": self.policy.reasoning_effort},
            "max_tokens": self.policy.max_tokens,
            "provider": {
                "require_parameters": self.policy.require_parameters,
                "allow_fallbacks": self.policy.allow_fallbacks,
            },
        }

    def _request(self, payload: dict) -> dict:
        last_error = None
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                response = self.transport.post(
                    ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.policy.timeout_seconds,
                )
                if response.status_code >= 400:
                    error = OpenRouterProviderError(
                        f"openrouter_audio_tags_http_{response.status_code}"
                    )
                    if (
                        response.status_code not in {408, 409, 429}
                        and response.status_code < 500
                    ):
                        raise error
                    last_error = error
                else:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise ValueError("response is not an object")
                    return data
            except httpx.TransportError:
                last_error = OpenRouterProviderError(
                    "openrouter_audio_tags_transport_error"
                )
            except (json.JSONDecodeError, ValueError):
                last_error = OpenRouterProviderError(
                    "openrouter_audio_tags_invalid_response"
                )
            if attempt < self.policy.max_attempts:
                self.sleeper(self.policy.retry_backoff_seconds)
        raise last_error or OpenRouterProviderError(
            "openrouter_audio_tags_request_failed"
        )

    def close(self):
        if self._owns_transport:
            self.transport.close()


def _parse_json_content(value: object) -> object:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("structured response content must be JSON text")
    content = value.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()
        if content.lower().startswith("json"):
            content = content[4:].lstrip()
    return json.loads(content)


def _response_shape(data: object) -> str:
    """Return safe categorical diagnostics without response content."""
    if not isinstance(data, dict):
        return "body_non_object"
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return "choices_invalid"
    choice = choices[0]
    if not isinstance(choice, dict):
        return "choice_non_object"
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "length", "content_filter", "tool_calls"}:
        finish_reason = "unknown"
    message = choice.get("message")
    if not isinstance(message, dict):
        return f"message_non_object_finish_{finish_reason}"
    content = message.get("content")
    if content is None:
        content_shape = "content_null"
    elif isinstance(content, str):
        content_shape = "content_text" if content.strip() else "content_empty"
    elif isinstance(content, list):
        content_shape = "content_blocks"
    elif isinstance(content, dict):
        content_shape = "content_object"
    else:
        content_shape = "content_other"
    reasoning_shape = (
        "reasoning_present" if message.get("reasoning") else "reasoning_absent"
    )
    return f"{content_shape}_finish_{finish_reason}_{reasoning_shape}"


def _usage(raw: Any, model: str) -> dict:
    value = raw or {}
    if not isinstance(value, dict):
        raise ValueError("usage must be an object")

    def integer(name):
        item = value.get(name, 0)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError("invalid usage")
        return item

    cost = value.get("cost", 0)
    if isinstance(cost, bool) or not isinstance(cost, int | float) or cost < 0:
        raise ValueError("invalid usage")
    return {
        "model": model,
        "in_tokens": integer("prompt_tokens"),
        "out_tokens": integer("completion_tokens"),
        "total_tokens": integer("total_tokens"),
        "cost_usd": cost,
    }
