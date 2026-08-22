from __future__ import annotations

import base64
import time

import httpx

from .audio import pcm16_mono_to_wav
from .errors import OpenRouterProviderError

ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"


class FishAudioClient:
    def __init__(self, policy, api_key: str, transport=None, sleeper=time.sleep):
        self.policy = policy
        self.api_key = api_key
        self.transport = transport or httpx.Client()
        self._owns_transport = transport is None
        self.sleeper = sleeper

    def synthesize(self, text: str, reference_audio: bytes) -> bytes:
        if not text or not reference_audio:
            raise ValueError("TTS input is incomplete")
        payload = self._payload(text, reference_audio)
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
                        f"openrouter_fish_audio_http_{response.status_code}"
                    )
                    if (
                        response.status_code not in {408, 409, 429}
                        and response.status_code < 500
                    ):
                        raise error
                    last_error = error
                elif response.content:
                    return pcm16_mono_to_wav(
                        response.content, sample_rate_hz=self.policy.sample_rate_hz
                    )
                else:
                    last_error = OpenRouterProviderError(
                        "openrouter_fish_audio_empty_response"
                    )
            except httpx.TransportError:
                last_error = OpenRouterProviderError(
                    "openrouter_fish_audio_transport_error"
                )
            if attempt < self.policy.max_attempts:
                self.sleeper(self.policy.retry_backoff_seconds)
        raise last_error or OpenRouterProviderError(
            "openrouter_fish_audio_request_failed"
        )

    def _payload(self, text: str, reference_audio: bytes) -> dict:
        return {
            "model": self.policy.model,
            "input": text,
            "input_references": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "data:audio/wav;base64,"
                        + base64.b64encode(reference_audio).decode("ascii")
                    },
                }
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

    def close(self):
        if self._owns_transport:
            self.transport.close()


def tts_text(tags: list[str], text: str) -> str:
    return " ".join([*tags, text])
