from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .prompt import build_response_schema, build_system_prompt, build_user_prompt

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    """A safe provider error with no request or response content."""


class OpenRouterClient:
    def __init__(self, policy, api_key: str, transport=None, sleeper=time.sleep):
        self.policy = policy
        self.api_key = api_key
        self.transport = transport or httpx.Client()
        self._owns_transport = transport is None
        self.sleeper = sleeper

    def extend(self, persona: dict, transcript: dict, dialogue_policy):
        payload = self._payload(persona, transcript, dialogue_policy)
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
                if response.status_code >= 400:
                    error = OpenRouterError(self._http_error_code(response))
                    if (
                        response.status_code in {408, 409, 429}
                        or response.status_code >= 500
                    ):
                        raise _Retryable(error)
                    raise error
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    raise _InvalidResponse("openrouter_response_not_json") from exc
                return self._parse_response(data)
            except _Retryable as exc:
                last_error = exc.error
            except httpx.TransportError:
                last_error = OpenRouterError("openrouter_transport_error")
            except _InvalidResponse as exc:
                last_error = OpenRouterError(exc.code)
            if attempt < self.policy.max_attempts:
                self.sleeper(self.policy.retry_backoff_seconds)
        raise last_error or OpenRouterError("openrouter_request_failed")

    def _payload(self, persona: dict, transcript: dict, dialogue_policy):
        schema = build_response_schema(dialogue_policy.min_utterances)
        return {
            "model": self.policy.model,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(dialogue_policy, schema),
                },
                {
                    "role": "user",
                    "content": build_user_prompt(persona, transcript),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "dialogue_extension",
                    "strict": True,
                    "schema": schema,
                },
            },
            "max_tokens": self.policy.max_tokens,
            "stream": False,
            "plugins": [{"id": "response-healing"}],
            "provider": {
                "require_parameters": self.policy.require_parameters,
                "allow_fallbacks": self.policy.allow_fallbacks,
            },
        }

    def _parse_response(self, data: object):
        if not isinstance(data, dict):
            raise _InvalidResponse("openrouter_response_shape_invalid")
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise _InvalidResponse("openrouter_choices_invalid")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _InvalidResponse("openrouter_choice_invalid")
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {None, "stop"}:
            code = {
                "length": "openrouter_completion_truncated",
                "content_filter": "openrouter_completion_filtered",
            }.get(finish_reason, "openrouter_completion_incomplete")
            raise _InvalidResponse(code)
        message = choice.get("message")
        if not isinstance(message, dict):
            raise _InvalidResponse("openrouter_message_invalid")
        if message.get("refusal"):
            raise _InvalidResponse("openrouter_completion_refused")
        content = message.get("content")
        if isinstance(content, dict):
            wire = content
        else:
            if isinstance(content, list) and len(content) == 1:
                part = content[0]
                if isinstance(part, dict) and part.get("type") in {
                    "text",
                    "output_text",
                }:
                    content = part.get("text")
            if not isinstance(content, str):
                raise _InvalidResponse("openrouter_structured_content_missing")
            try:
                wire = json.loads(content)
            except json.JSONDecodeError as exc:
                raise _InvalidResponse(
                    "openrouter_structured_content_not_json"
                ) from exc
        if not isinstance(wire, dict):
            raise _InvalidResponse("openrouter_structured_content_invalid")
        raw_usage = data.get("usage") or {}
        if not isinstance(raw_usage, dict):
            raise _InvalidResponse("openrouter_usage_invalid")
        try:
            usage = {
                "model": self.policy.model,
                "in_tokens": _usage_integer(raw_usage.get("prompt_tokens")),
                "out_tokens": _usage_integer(raw_usage.get("completion_tokens")),
                "total_tokens": _usage_integer(raw_usage.get("total_tokens")),
                "cost_usd": _usage_number(raw_usage.get("cost")),
            }
        except (TypeError, ValueError) as exc:
            raise _InvalidResponse("openrouter_usage_invalid") from exc
        return wire, usage

    @staticmethod
    def _http_error_code(response) -> str:
        status = response.status_code
        if status != 400:
            return f"openrouter_http_{status}"
        try:
            body = response.json()
            error = body.get("error") if isinstance(body, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            metadata = error.get("metadata") if isinstance(error, dict) else None
            raw = metadata.get("raw") if isinstance(metadata, dict) else None
            if isinstance(raw, str):
                try:
                    provider_body = json.loads(raw)
                    provider_error = (
                        provider_body.get("error")
                        if isinstance(provider_body, dict)
                        else None
                    )
                    provider_message = (
                        provider_error.get("message")
                        if isinstance(provider_error, dict)
                        else None
                    )
                    if isinstance(provider_message, str):
                        message = " ".join(
                            part
                            for part in (message, provider_message)
                            if isinstance(part, str)
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            message = None
        if not isinstance(message, str):
            return "openrouter_http_400"
        normalized = message.lower()
        if "schema" in normalized:
            return "openrouter_http_400_invalid_schema"
        if "plugin" in normalized or "response-healing" in normalized:
            return "openrouter_http_400_plugin_rejected"
        if "parameter" in normalized or "unsupported" in normalized:
            return "openrouter_http_400_unsupported_parameter"
        if "context" in normalized or "token" in normalized:
            return "openrouter_http_400_context_limit"
        return "openrouter_http_400"

    def close(self):
        if self._owns_transport:
            self.transport.close()


class _Retryable(Exception):
    def __init__(self, error: OpenRouterError):
        self.error = error


class _InvalidResponse(Exception):
    def __init__(self, code: str):
        self.code = code


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
