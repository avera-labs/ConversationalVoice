from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

EMOTIONS = ["angry", "sad", "happy", "surprised", "neutral"]
TONES = ["aggressive", "warm", "cold", "nervous", "calm", "playful"]
LEVELS = ["low", "medium", "high"]


def build_response_schema(speaker_ids: Sequence[int]) -> dict[str, Any]:
    ids = sorted(str(value) for value in speaker_ids)

    def nullable_string(description: str) -> dict[str, Any]:
        return {
            "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
            "description": description,
        }

    speaker_properties = {
        "name": nullable_string("Speaker name if identifiable, otherwise null."),
        "age": nullable_string("Estimated age range such as '30s', otherwise null."),
        "ethnicity": nullable_string("Perceived ethnicity, otherwise null."),
        "gender": nullable_string("Perceived gender, otherwise null."),
        "tag": {
            "type": "string",
            "minLength": 1,
            "description": "One first-person actor instruction under 30 words.",
        },
        "alpha": {
            "type": "string",
            "enum": LEVELS,
            "description": "Distinctiveness and consistency of the vocal persona.",
        },
        "evidence": nullable_string(
            "Direct quote and behavioral observation for high or medium alpha; null for low."
        ),
        "primary_emotion": {
            "type": "string",
            "enum": EMOTIONS,
            "description": "Dominant emotion audible in the voice.",
        },
        "secondary_emotion": {
            "anyOf": [
                {"type": "string", "enum": EMOTIONS},
                {"type": "null"},
            ],
            "description": "Less dominant audible emotion, otherwise null.",
        },
        "emotion_intensity": {
            "type": "string",
            "enum": LEVELS,
            "description": "Strength of audible emotional expression.",
        },
        "laugh": {"type": "boolean", "description": "Clearly audible laughter."},
        "cry": {"type": "boolean", "description": "Clearly audible crying."},
        "whisper": {"type": "boolean", "description": "Clearly audible whispering."},
        "shout": {"type": "boolean", "description": "Clearly audible shouting."},
        "sigh": {"type": "boolean", "description": "Clearly audible sighing."},
        "overall_tone": {
            "type": "string",
            "enum": TONES,
            "description": "General quality of this speaker's vocal delivery.",
        },
    }
    speaker = {
        "type": "object",
        "properties": speaker_properties,
        "required": list(speaker_properties),
        "additionalProperties": False,
    }
    scene_properties = {
        "description": {
            "type": "string",
            "minLength": 1,
            "description": "One stage-setting sentence without speaker IDs.",
        },
        "overall_tone": {
            "type": "string",
            "enum": TONES,
            "description": "Dominant audible tone of the whole conversation.",
        },
        "emotion_intensity": {
            "type": "string",
            "enum": LEVELS,
            "description": "Overall audible emotional intensity.",
        },
    }
    root_properties = {
        "scene": {
            "type": "object",
            "properties": scene_properties,
            "required": list(scene_properties),
            "additionalProperties": False,
        },
        "speakers": {
            "type": "object",
            "properties": {speaker_id: speaker for speaker_id in ids},
            "required": ids,
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": root_properties,
        "required": list(root_properties),
        "additionalProperties": False,
    }


def build_system_prompt(
    speaker_ids: Sequence[int], schema: dict[str, Any] | None = None
) -> str:
    ids = sorted(str(value) for value in speaker_ids)
    rendered = ", ".join(ids)
    schema = schema or build_response_schema(speaker_ids)
    serialized_schema = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"""You analyze vocal personas from conversation audio and its transcript.

Use the audio as the primary signal for tone, energy, rhythm, vocal style,
emotion, and audible events. Use the transcript only to understand words and
speaker identity. Do not infer an emotion or event merely from the topic.
Listen to the complete audio.

Return exactly these transcript speaker IDs as object keys: {rendered}. Never
renumber, omit, merge, or invent speakers. Nullable fields are required and
must be JSON null when unknown.

Scene rules:
- description is one stage-setting sentence and never mentions speaker IDs;
- tone and intensity describe how the conversation sounds, not its topic;
- low intensity stays even, medium has audible shifts, and high stays charged.

Speaker rules:
- primary and secondary emotion, intensity, tone, and event flags come only
  from audible prosody, pitch, pace, vocal quality, or the event itself;
- the tag is one first-person actor instruction under 30 words;
- high alpha means an unmistakable persona that dominates the clip;
- medium alpha means a recognizable but inconsistent persona;
- low alpha means no distinct persona and uses the exact fallback tag
  \"You enjoy having a good conversation.\";
- evidence is null for low alpha; for high or medium it includes one direct
  quote and one behavioral observation, and medium also notes what dilutes it.

Your response is governed by OpenRouter strict structured output. Return only
the JSON value described by this exact JSON Schema:
{serialized_schema}"""
