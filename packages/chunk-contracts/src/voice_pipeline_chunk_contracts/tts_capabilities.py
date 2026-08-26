"""Model-to-capability mapping for canonical utterance synthesis inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping


logger = logging.getLogger(__name__)

TTS_MODEL_CAPABILITIES = {
    "fish-audio/s2.1-pro": frozenset({"text_with_audio_tags"}),
    # "mimo-v2.5-tts-voiceclone": frozenset({"instruction"}),
}
DEFAULT_TTS_CAPABILITIES = frozenset({"text"})


@dataclass(frozen=True, slots=True)
class TtsInputs:
    text: str
    instruction: str | None


def tts_capabilities(model: str) -> frozenset[str]:
    capabilities = TTS_MODEL_CAPABILITIES.get(model)
    if capabilities is None:
        logger.warning(
            "TTS model %r is missing from TTS_MODEL_CAPABILITIES; "
            "using plain text. Add the model to the mapping to declare its features.",
            model,
        )
        return DEFAULT_TTS_CAPABILITIES
    return capabilities


def select_tts_inputs(model: str, utterance: Mapping[str, object]) -> TtsInputs:
    capabilities = tts_capabilities(model)
    if "text_with_audio_tags" in capabilities:
        text = utterance.get("text_with_audio_tags")
    else:
        text = utterance.get("text")
    instruction = (
        utterance.get("instruction") if "instruction" in capabilities else None
    )
    if not isinstance(text, str) or not text:
        raise ValueError("TTS text is unavailable for model capabilities")
    if instruction is not None and (
        not isinstance(instruction, str) or not instruction.strip()
    ):
        raise ValueError("TTS instruction is invalid")
    return TtsInputs(text, instruction)
