from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    size_bytes: int
    sha256: str


def build_persona_document(wire, usage, speaker_mapping, policy):
    return {
        "scene": wire["scene"],
        "speakers": [
            {**wire["speakers"][speaker_id], "speaker_id": speaker_id}
            for speaker_id in sorted(wire["speakers"])
        ],
        "usage": usage,
        "schema_version": 1,
        "backend": "openrouter",
        "config_version": policy.config_version,
        "language": "en",
        "speaker_mapping": [
            {"output_slot": slot, "diarization_speaker_id": speaker_id}
            for slot, speaker_id in enumerate(speaker_mapping)
        ],
    }


def write_canonical_json(document: dict, path: Path) -> ArtifactMetadata:
    data = canonical_json_bytes(document)
    path.write_bytes(data)
    return ArtifactMetadata(len(data), hashlib.sha256(data).hexdigest())


def canonical_json_bytes(document: dict) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
