from __future__ import annotations

import base64
import re
import time
import unicodedata
from dataclasses import dataclass

import requests

from .errors import ScoringError

DEFAULT_ASR_MODEL = "qwen/qwen3-asr-1.7b"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    model: str
    generation_id: str | None
    usage: dict[str, object]


class OpenRouterAsrClient:
    """CPU-independent ASR transport using OpenRouter's dedicated STT endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_ASR_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key.strip() or "/" not in model:
            raise ValueError("invalid OpenRouter ASR configuration")
        if timeout_seconds <= 0 or max_attempts <= 0:
            raise ValueError("invalid OpenRouter ASR retry policy")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.session = session or requests.Session()

    def transcribe(self, audio: bytes, *, language: str) -> TranscriptionResult:
        if not audio:
            raise ScoringError("asr_empty_audio")
        request = {
            "model": self.model,
            "input_audio": {
                "data": base64.b64encode(audio).decode("ascii"),
                "format": "wav",
            },
            "language": language.split("-", 1)[0].lower(),
            "temperature": 0,
            "provider": {"data_collection": "deny", "zdr": True},
        }
        response = None
        for attempt in range(self.max_attempts):
            try:
                response = self.session.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request,
                    timeout=self.timeout_seconds,
                )
                if response.status_code < 500 and response.status_code != 429:
                    break
            except requests.RequestException:
                if attempt + 1 == self.max_attempts:
                    raise ScoringError("openrouter_asr_request_failed") from None
            if attempt + 1 < self.max_attempts:
                time.sleep(2**attempt)
        if response is None or not 200 <= response.status_code < 300:
            raise ScoringError("openrouter_asr_request_failed")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ScoringError("openrouter_asr_invalid_response") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        if not isinstance(text, str) or not isinstance(usage, dict):
            raise ScoringError("openrouter_asr_invalid_response")
        return TranscriptionResult(
            text=text.strip(),
            model=self.model,
            generation_id=response.headers.get("X-Generation-Id"),
            usage=usage,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "provider": "openrouter",
            "endpoint": self.endpoint,
            "model": self.model,
            "audio_format": "wav",
            "local_model_weights": False,
            "provider_policy": {"data_collection": "deny", "zdr": True},
        }

    def close(self) -> None:
        self.session.close()


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def error_rate(reference: str, hypothesis: str, *, language: str) -> dict[str, object]:
    normalized_reference = _normalized_text(reference)
    normalized_hypothesis = _normalized_text(hypothesis)
    if language.split("-", 1)[0].lower() == "zh":
        unit = "character"
        reference_units = [
            value
            for value in normalized_reference
            if not value.isspace() and not unicodedata.category(value).startswith("P")
        ]
        hypothesis_units = [
            value
            for value in normalized_hypothesis
            if not value.isspace() and not unicodedata.category(value).startswith("P")
        ]
        metric = "cer"
    else:
        unit = "word"
        reference_units = re.findall(r"[\w]+(?:['’][\w]+)?", normalized_reference)
        hypothesis_units = re.findall(r"[\w]+(?:['’][\w]+)?", normalized_hypothesis)
        metric = "wer"
    if not reference_units:
        raise ScoringError("asr_empty_reference")
    substitutions, deletions, insertions = _levenshtein_counts(
        reference_units, hypothesis_units
    )
    edits = substitutions + deletions + insertions
    return {
        "metric": metric,
        "unit": unit,
        "value": edits / len(reference_units),
        "edit_count": edits,
        "reference_unit_count": len(reference_units),
        "hypothesis_unit_count": len(hypothesis_units),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "normalized_reference": (
            "".join(reference_units) if metric == "cer" else " ".join(reference_units)
        ),
        "normalized_hypothesis": (
            "".join(hypothesis_units) if metric == "cer" else " ".join(hypothesis_units)
        ),
    }


def _levenshtein_counts(
    reference: list[str], hypothesis: list[str]
) -> tuple[int, int, int]:
    costs = [list(range(len(hypothesis) + 1))]
    operations = [["insert"] * (len(hypothesis) + 1)]
    operations[0][0] = "equal"
    for reference_index, reference_value in enumerate(reference, start=1):
        row = [reference_index]
        row_operations = ["delete"]
        for hypothesis_index, hypothesis_value in enumerate(hypothesis, start=1):
            if reference_value == hypothesis_value:
                row.append(costs[reference_index - 1][hypothesis_index - 1])
                row_operations.append("equal")
                continue
            candidates = (
                (costs[reference_index - 1][hypothesis_index - 1] + 1, "substitute"),
                (costs[reference_index - 1][hypothesis_index] + 1, "delete"),
                (row[hypothesis_index - 1] + 1, "insert"),
            )
            cost, operation = min(candidates, key=lambda value: value[0])
            row.append(cost)
            row_operations.append(operation)
        costs.append(row)
        operations.append(row_operations)
    substitutions = deletions = insertions = 0
    reference_index = len(reference)
    hypothesis_index = len(hypothesis)
    while reference_index or hypothesis_index:
        operation = operations[reference_index][hypothesis_index]
        if operation == "equal":
            reference_index -= 1
            hypothesis_index -= 1
        elif operation == "substitute":
            substitutions += 1
            reference_index -= 1
            hypothesis_index -= 1
        elif operation == "delete":
            deletions += 1
            reference_index -= 1
        else:
            insertions += 1
            hypothesis_index -= 1
    return substitutions, deletions, insertions
