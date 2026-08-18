from __future__ import annotations

import base64
import json
import time

import httpx

from .audio import pcm16_mono_to_wav

ASR_ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
TTS_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"


class OpenRouterFishAudioError(RuntimeError):
    """A safe provider error with no audio, transcript, or response content."""


class OpenRouterFishAudioClient:
    def __init__(self, policy, api_key: str, transport=None, sleeper=time.sleep):
        self.policy = policy
        self.api_key = api_key
        self.transport = transport or httpx.Client()
        self._owns_transport = transport is None
        self.sleeper = sleeper

    def transcribe_reference(self, audio: bytes) -> str:
        if not audio:
            raise ValueError("reference audio is empty")

        def request():
            return self.transport.post(
                ASR_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.policy.transcription_model,
                    "input_audio": {
                        "data": base64.b64encode(audio).decode("ascii"),
                        "format": "wav",
                    },
                    "language": "en",
                },
                timeout=self.policy.timeout_seconds,
            )

        response = self._request(request)
        try:
            data = response.json()
            text = data["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty transcript")
            return " ".join(text.split())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise OpenRouterFishAudioError(
                "openrouter_fish_audio_invalid_asr_response"
            ) from exc

    def synthesize(
        self, text: str, reference_audio: bytes, reference_text: str
    ) -> bytes:
        if not text or not reference_audio or not reference_text:
            raise ValueError("TTS input is incomplete")
        payload = {
            "model": self.policy.model,
            "input": text,
            "input_references": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "data:audio/wav;base64,"
                        + base64.b64encode(reference_audio).decode("ascii")
                    },
                },
                {"type": "text", "text": reference_text},
            ],
            "response_format": "pcm",
            "provider": {
                "options": {
                    "fish-audio": {
                        "temperature": self.policy.temperature,
                        "top_p": self.policy.top_p,
                        "prosody": {
                            "speed": 1.0,
                            "volume": 0.0,
                            "normalize_loudness": self.policy.normalize_loudness,
                        },
                        "chunk_length": self.policy.chunk_length,
                        "min_chunk_length": self.policy.min_chunk_length,
                        "normalize": self.policy.normalize_text,
                        "sample_rate": self.policy.sample_rate_hz,
                        "latency": self.policy.latency,
                        "max_new_tokens": self.policy.max_new_tokens,
                        "repetition_penalty": self.policy.repetition_penalty,
                        "condition_on_previous_chunks": True,
                        "early_stop_threshold": 1.0,
                    }
                }
            },
        }

        def request():
            return self.transport.post(
                TTS_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.policy.timeout_seconds,
            )

        response = self._request(request)
        if not response.content:
            raise OpenRouterFishAudioError("openrouter_fish_audio_empty_tts_response")
        try:
            return pcm16_mono_to_wav(
                response.content, sample_rate_hz=self.policy.sample_rate_hz
            )
        except ValueError as exc:
            raise OpenRouterFishAudioError(
                "openrouter_fish_audio_invalid_tts_response"
            ) from exc

    def _request(self, request):
        last_error: Exception | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                response = request()
                if response.status_code >= 400:
                    error = OpenRouterFishAudioError(
                        f"openrouter_fish_audio_http_{response.status_code}"
                    )
                    if (
                        response.status_code in {408, 409, 429}
                        or response.status_code >= 500
                    ):
                        raise _Retryable(error)
                    raise error
                return response
            except _Retryable as exc:
                last_error = exc.error
            except httpx.TransportError:
                last_error = OpenRouterFishAudioError(
                    "openrouter_fish_audio_transport_error"
                )
            if attempt < self.policy.max_attempts:
                self.sleeper(self.policy.retry_backoff_seconds)
        raise last_error or OpenRouterFishAudioError(
            "openrouter_fish_audio_request_failed"
        )

    def close(self):
        if self._owns_transport:
            self.transport.close()


class _Retryable(Exception):
    def __init__(self, error: OpenRouterFishAudioError):
        self.error = error


def tts_text(utterance: dict) -> str:
    """Render canonical audio tags and spoken text for Fish Audio or Eleven v3."""

    pieces = [*utterance["audio_tags"]]
    if utterance["text"]:
        pieces.append(utterance["text"])
    return " ".join(pieces)
