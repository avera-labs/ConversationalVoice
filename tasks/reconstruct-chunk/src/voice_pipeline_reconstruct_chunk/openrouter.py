from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
from voice_pipeline_chunk_contracts import (
    AUDIO_TAGS,
    TaggedTextError,
    parse_text_with_audio_tags,
)

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
        correction = None
        last_error = None
        for attempt in range(1, self.policy.max_attempts + 1):
            data = None
            payload = self._payload(audio, text, correction=correction)
            try:
                data = self._request(payload)
                return self._parse_annotation(data, text)
            except _Retryable as exc:
                last_error = exc.error
                correction = _correction(exc.error.args[0])
            except httpx.TransportError:
                last_error = OpenRouterProviderError(
                    "openrouter_audio_tags_transport_error"
                )
                correction = _correction("openrouter_audio_tags_transport_error")
            except _CorrectionNeeded as exc:
                last_error = OpenRouterProviderError(
                    "openrouter_audio_tags_" + exc.code
                )
                correction = {
                    "reason_code": exc.code,
                    "requirement": exc.requirement,
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                code = "invalid_response_" + _response_shape(locals().get("data"))
                last_error = OpenRouterProviderError("openrouter_audio_tags_" + code)
                correction = _correction(code)
            if attempt < self.policy.max_attempts:
                self.sleeper(self.policy.retry_backoff_seconds)
        raise last_error or OpenRouterProviderError(
            "openrouter_audio_tags_request_failed"
        )

    def _parse_annotation(self, data: dict, source_text: str) -> tuple[dict, dict]:
        choices = data["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise _CorrectionNeeded(
                "response_shape_invalid",
                "Return exactly one structured-output choice.",
            )
        wire = _parse_json_content(choices[0]["message"]["content"])
        if not isinstance(wire, dict) or set(wire) != {
            "text_with_audio_tags",
            "instruction",
        }:
            raise _CorrectionNeeded(
                "response_shape_invalid",
                "Return only text_with_audio_tags and instruction.",
            )
        try:
            tagged = parse_text_with_audio_tags(wire["text_with_audio_tags"])
        except TaggedTextError as exc:
            raise _CorrectionNeeded(exc.code, exc.requirement) from exc
        if tagged.text != source_text:
            raise _CorrectionNeeded(
                "derived_text_mismatch",
                "Removing approved audio tags must reproduce ANNOTATE_THIS_EXACT_TEXT exactly.",
            )
        instruction = wire["instruction"]
        if (
            not isinstance(instruction, str)
            or not instruction
            or instruction != instruction.strip()
            or "[" in instruction
            or "]" in instruction
        ):
            raise _CorrectionNeeded(
                "instruction_invalid",
                "instruction must be one non-empty concise sentence with no square-bracket tags.",
            )
        usage = _usage(data.get("usage"), self.policy.model)
        return {
            "text": tagged.text,
            "text_with_audio_tags": tagged.text_with_audio_tags,
            "instruction": instruction,
        }, usage

    def _payload(
        self, audio: bytes, text: str, correction: dict[str, str] | None = None
    ) -> dict:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text_with_audio_tags": {
                    "type": "string",
                    "description": (
                        "The exact supplied text with approved audio tags inserted "
                        "at their audible positions."
                    ),
                },
                "instruction": {
                    "type": "string",
                    "description": "One concise actor-facing performance instruction.",
                },
            },
            "required": ["text_with_audio_tags", "instruction"],
        }
        schema_json = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        allowed_tags = json.dumps(
            sorted(AUDIO_TAGS), ensure_ascii=False, separators=(",", ":")
        )
        user_text = "ANNOTATE_THIS_EXACT_TEXT:\n" + text
        if correction is not None:
            user_text += (
                "\n\nCORRECTION_REQUIRED:\n"
                "The previous response was rejected.\n"
                f"reason_code: {correction['reason_code']}\n"
                f"requirement: {correction['requirement']}\n"
                "Regenerate the complete JSON object from scratch."
            )
        return {
            "model": self.policy.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Analyze only the supplied utterance audio and annotate the supplied "
                        "canonical text without rewriting it. Output exactly "
                        "text_with_audio_tags and instruction; never output text. Insert only "
                        "approved audio tags at the exact audible position before, within, or "
                        "after words. Removing all tags must reproduce the supplied text "
                        "exactly, including wording, order, case, and punctuation. A tag may "
                        "repeat at different positions. One or two adjacent tags are allowed "
                        "at one position; never use three adjacent tags. Square brackets are "
                        "reserved for approved tags. instruction must be one non-empty, "
                        "concise actor-facing sentence grounded in audible delivery; do not "
                        "repeat the dialogue or include tags. Do not infer inaudible events. "
                        "Return only one JSON object with no Markdown or commentary.\n\n"
                        "APPROVED_AUDIO_TAGS:\n"
                        + allowed_tags
                        + "\n\nJSON_SCHEMA:\n"
                        + schema_json
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
                        {"type": "text", "text": user_text},
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
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                raise _Retryable(error)
            raise error
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("response is not an object")
        return data

    def close(self):
        if self._owns_transport:
            self.transport.close()


class _Retryable(Exception):
    def __init__(self, error: OpenRouterProviderError):
        self.error = error


class _CorrectionNeeded(Exception):
    def __init__(self, code: str, requirement: str):
        self.code = code
        self.requirement = requirement


def _correction(code: str) -> dict[str, str]:
    return {
        "reason_code": code,
        "requirement": (
            "Return one complete JSON object that follows the schema and "
            "exact-text constraints."
        ),
    }


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
