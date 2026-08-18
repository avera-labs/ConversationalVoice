from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx

from .prompt import build_response_schema, build_system_prompt

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    """A safe provider error that contains no request or response content."""


class OpenRouterClient:
    def __init__(self, policy, api_key: str, transport=None, sleeper=time.sleep):
        self.policy = policy
        self.api_key = api_key
        self.transport = transport or httpx.Client()
        self._owns_transport = transport is None
        self.sleeper = sleeper

    def analyze(self, mp3: bytes, srt: str, speaker_mapping: tuple[int, int]):
        payload = self._payload(mp3, srt, speaker_mapping)
        last_error: Exception | None = None
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
                status = response.status_code
                if status >= 400:
                    error = OpenRouterError(f"openrouter_http_{status}")
                    if status not in {408, 409, 429} and status < 500:
                        raise error
                    raise _Retryable(error)
                data = response.json()
                wire, usage = self._parse_response(data)
                return wire, usage
            except _Retryable as exc:
                last_error = exc.error
            except httpx.TransportError:
                last_error = OpenRouterError("openrouter_transport_error")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                last_error = OpenRouterError("openrouter_invalid_response")
            if attempt < self.policy.max_attempts:
                self.sleeper(self.policy.retry_backoff_seconds)
        raise last_error or OpenRouterError("openrouter_request_failed")

    def _payload(self, mp3: bytes, srt: str, speaker_mapping: tuple[int, int]):
        schema = build_response_schema(speaker_mapping)
        return {
            "model": self.policy.model,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(speaker_mapping, schema),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(mp3).decode("ascii"),
                                "format": "mp3",
                            },
                        },
                        {"type": "text", "text": f"TRANSCRIPT:\n{srt}"},
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "persona_analysis",
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

    def _parse_response(self, data: object):
        if not isinstance(data, dict):
            raise TypeError("response must be an object")
        choices = data["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("response must contain one choice")
        message = choices[0]["message"]
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("structured response content must be JSON text")
        wire = json.loads(content)
        raw_usage = data.get("usage") or {}
        if not isinstance(raw_usage, dict):
            raise TypeError("usage must be an object")
        usage = {
            "model": self.policy.model,
            "in_tokens": _usage_integer(raw_usage.get("prompt_tokens")),
            "out_tokens": _usage_integer(raw_usage.get("completion_tokens")),
            "total_tokens": _usage_integer(raw_usage.get("total_tokens")),
            "cost_usd": _usage_number(raw_usage.get("cost")),
        }
        return wire, usage

    def close(self):
        if self._owns_transport:
            self.transport.close()


class _Retryable(Exception):
    def __init__(self, error: OpenRouterError):
        self.error = error


def _usage_integer(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid usage integer")
    return value


def _usage_number(value: Any) -> int | float:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError("invalid usage number")
    return value
