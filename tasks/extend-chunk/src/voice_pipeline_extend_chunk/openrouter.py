from __future__ import annotations

import json
import time
from typing import Any

import httpx
from voice_pipeline_chunk_contracts import TaggedTextError, parse_text_with_audio_tags

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

    def extend(
        self, persona: dict, transcript: dict, dialogue_policy, language: str = "en"
    ):
        last_error: Exception | None = None
        correction = None
        for attempt in range(1, self.policy.max_attempts + 1):
            payload = self._payload(
                persona, transcript, dialogue_policy, language, correction=correction
            )
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
                wire, usage = self._parse_response(data)
                return _normalize_wire(wire, dialogue_policy), usage
            except _Retryable as exc:
                last_error = exc.error
                correction = _correction(exc.error.args[0], "response")
            except httpx.TransportError:
                last_error = OpenRouterError("openrouter_transport_error")
                correction = _correction("openrouter_transport_error", "response")
            except _InvalidResponse as exc:
                last_error = OpenRouterError(exc.code)
                correction = _correction(exc.code, "response")
            except _CorrectionNeeded as exc:
                last_error = OpenRouterError(exc.code)
                correction = {
                    "reason_code": exc.code,
                    "location": exc.location,
                    "requirement": exc.requirement,
                }
            if attempt < self.policy.max_attempts:
                self.sleeper(self.policy.retry_backoff_seconds)
        raise last_error or OpenRouterError("openrouter_request_failed")

    def _payload(
        self,
        persona: dict,
        transcript: dict,
        dialogue_policy,
        language: str,
        correction: dict[str, str] | None = None,
    ):
        schema = build_response_schema(dialogue_policy.min_utterances)
        return {
            "model": self.policy.model,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        dialogue_policy, schema, language=language
                    ),
                },
                {
                    "role": "user",
                    "content": build_user_prompt(
                        persona, transcript, language=language, correction=correction
                    ),
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


class _CorrectionNeeded(Exception):
    def __init__(self, code: str, location: str, requirement: str):
        self.code = code
        self.location = location
        self.requirement = requirement


_WIRE_FIELDS = {
    "utterance_index",
    "speaker_id",
    "text_with_audio_tags",
    "instruction",
    "type",
    "placement",
}


def _normalize_wire(wire: object, policy) -> dict:
    if not isinstance(wire, dict) or set(wire) != {"utterances"}:
        raise _CorrectionNeeded(
            "response_shape_invalid",
            "response",
            "Return exactly one object containing only the utterances array.",
        )
    utterances = wire["utterances"]
    if (
        not isinstance(utterances, list)
        or not policy.min_utterances <= len(utterances) <= policy.max_utterances
    ):
        raise _CorrectionNeeded(
            "utterance_count_invalid",
            "utterances",
            f"Return between {policy.min_utterances} and {policy.max_utterances} utterances.",
        )
    normalized = []
    for index, raw in enumerate(utterances):
        location = f"utterance_index={index}"
        if not isinstance(raw, dict):
            raise _CorrectionNeeded(
                "response_shape_invalid", location, "Every utterance must be an object."
            )
        if "text" in raw:
            raise _CorrectionNeeded(
                "forbidden_text_field",
                location,
                "Do not output text; output text_with_audio_tags and let the application derive text.",
            )
        missing = _WIRE_FIELDS - set(raw)
        if missing:
            raise _CorrectionNeeded(
                "required_field_missing",
                location,
                "Include every required utterance field.",
            )
        if set(raw) != _WIRE_FIELDS:
            raise _CorrectionNeeded(
                "response_shape_invalid",
                location,
                "Do not include fields outside the required utterance schema.",
            )
        if (
            isinstance(raw["utterance_index"], bool)
            or not isinstance(raw["utterance_index"], int)
            or raw["utterance_index"] != index
        ):
            raise _CorrectionNeeded(
                "utterance_index_invalid",
                location,
                "utterance_index must start at zero and increase by one without gaps.",
            )
        try:
            tagged = parse_text_with_audio_tags(raw["text_with_audio_tags"])
        except TaggedTextError as exc:
            raise _CorrectionNeeded(exc.code, location, exc.requirement) from exc
        instruction = raw["instruction"]
        if (
            not isinstance(instruction, str)
            or not instruction
            or instruction != instruction.strip()
            or "[" in instruction
            or "]" in instruction
        ):
            raise _CorrectionNeeded(
                "instruction_invalid",
                location,
                "instruction must be one non-empty concise sentence with no square-bracket tags.",
            )
        utterance_type = raw["type"]
        placement = raw["placement"]
        speaker_id = raw["speaker_id"]
        if (
            isinstance(speaker_id, bool)
            or not isinstance(speaker_id, int)
            or speaker_id not in {0, 1}
            or utterance_type not in {"dialogue", "backchannel", "paralinguistic"}
            or placement not in {"sequential", "overlap_previous"}
            or (utterance_type == "dialogue" and placement != "sequential")
            or (utterance_type != "paralinguistic" and not tagged.text)
            or (utterance_type == "paralinguistic" and not tagged.tags)
        ):
            raise _CorrectionNeeded(
                "speaker_or_placement_invalid",
                location,
                "Use a valid speaker, type, placement, and non-empty derived dialogue text.",
            )
        normalized.append({**raw, "text": tagged.text})
    if {item["speaker_id"] for item in normalized} != {0, 1}:
        raise _CorrectionNeeded(
            "speaker_or_placement_invalid",
            "utterances",
            "Both fixed speakers 0 and 1 must appear.",
        )
    if normalized[0]["placement"] != "sequential":
        raise _CorrectionNeeded(
            "speaker_or_placement_invalid",
            "utterance_index=0",
            "The first utterance must be sequential.",
        )
    for previous, current in zip(normalized, normalized[1:]):
        if (
            current["placement"] == "overlap_previous"
            and current["speaker_id"] == previous["speaker_id"]
        ):
            raise _CorrectionNeeded(
                "speaker_or_placement_invalid",
                f"utterance_index={current['utterance_index']}",
                "A speaker must not overlap its own previous utterance.",
            )
    return {"utterances": normalized}


def _correction(code: str, location: str) -> dict[str, str]:
    return {
        "reason_code": code,
        "location": location,
        "requirement": "Return one complete JSON object that follows the supplied schema and instructions.",
    }


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
