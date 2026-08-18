from __future__ import annotations

import json
from typing import Any

from voice_pipeline_chunk_contracts import AUDIO_TAGS


def build_response_schema(min_utterances: int) -> dict[str, Any]:
    utterance_properties = {
        "utterance_index": {"type": "integer", "minimum": 0},
        "speaker_id": {"type": "integer", "minimum": 0, "maximum": 1},
        "text": {
            "type": "string",
            "description": "Spoken words only; empty only for a pure paralinguistic event.",
        },
        "tone": {
            "type": "string",
            "description": "A non-empty, concise actor-facing delivery description.",
        },
        "type": {
            "type": "string",
            "enum": ["dialogue", "backchannel", "paralinguistic"],
        },
        "placement": {
            "type": "string",
            "enum": ["sequential", "overlap_previous"],
        },
        "audio_tags": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(AUDIO_TAGS)},
            "maxItems": 3,
        },
    }
    return {
        "type": "object",
        "properties": {
            "utterances": {
                "type": "array",
                "minItems": min_utterances,
                "items": {
                    "type": "object",
                    "properties": utterance_properties,
                    "required": list(utterance_properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["utterances"],
        "additionalProperties": False,
    }


def build_system_prompt(policy, schema: dict[str, Any]) -> str:
    return f"""You write a continuation of an existing natural two-person conversation.

Continue from the final source utterance without repeating, summarizing, correcting, or
rewriting the source. Preserve the topic, facts, relationship, language, and the two speaker
personas. Speaker IDs 0 and 1 are fixed identities and must never be swapped.

Aim for approximately {policy.target_duration_seconds} seconds or {policy.target_words}
spoken words. Return between {policy.min_utterances} and {policy.max_utterances} utterances.
Both speakers must appear. Keep turns conversational rather than speech-like monologues.

Every utterance has exactly one type:
- dialogue: ordinary spoken content; it must be sequential;
- backchannel: a brief acknowledgement such as "yeah", "right", or "mm-hmm";
- paralinguistic: a vocal event such as laughter, crying, a sigh, breath, or cough.

Use overlap_previous only for a short backchannel or paralinguistic event by the speaker who
did not produce the previous utterance. The first utterance must be sequential. Add reactions
sparingly and only where they improve a believable exchange.

text contains spoken words only and must not contain square-bracket tags. For a pure
paralinguistic event text may be empty. audio_tags contains only tags allowed by the schema.
Do not repeat an audio tag within one utterance. tone must never be empty.
These tags use a shared Fish Audio S2.1 Pro / Eleven v3 square-bracket convention. tone is a
short performance direction and is not a second dialogue field. When a supported audio tag
can express the requested tone, include that tag so the synthesis request carries the audible
direction as well as the metadata.

utterance_index starts at zero and increases by one without gaps.

The response is governed by strict structured output. Return only JSON matching this schema:
{json.dumps(schema, sort_keys=True, separators=(",", ":"))}
"""


def build_user_prompt(persona: dict, transcript: dict) -> str:
    output_speaker_identity = [
        {
            "speaker_id": item["output_slot"],
            "persona_speaker_id": str(item["diarization_speaker_id"]),
        }
        for item in persona["speaker_mapping"]
    ]
    inputs = {
        "output_speaker_identity": output_speaker_identity,
        "persona": persona,
        "transcript": transcript,
    }
    return (
        "Continue this conversation from the supplied canonical inputs:\n"
        + json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    )
