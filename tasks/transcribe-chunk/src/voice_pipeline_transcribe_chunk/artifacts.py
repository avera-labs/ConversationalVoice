from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .utterances import Utterance, Word


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    size_bytes: int
    sha256: str


def model_identity(policy) -> dict[str, object]:
    return {
        "repo_id": policy.model.repo_id,
        "revision": policy.model.revision,
        "config_version": policy.config_version,
    }


def build_artifacts(*, policy, speaker_outputs):
    shared = {
        "schema_version": 1,
        "backend": "parakeet_tdt",
        "model": model_identity(policy),
        "language": "en",
        "timebase": "chunk",
    }
    transcript_speakers = []
    word_speakers = []
    for slot, speaker_id, words, utterances in speaker_outputs:
        transcript_speakers.append(
            {
                "output_slot": slot,
                "diarization_speaker_id": speaker_id,
                "utterances": [
                    _utterance(index, item) for index, item in enumerate(utterances)
                ],
            }
        )
        word_speakers.append(
            {
                "output_slot": slot,
                "diarization_speaker_id": speaker_id,
                "words": [_word(index, item) for index, item in enumerate(words)],
            }
        )
    return (
        {**shared, "speakers": transcript_speakers},
        {**shared, "speakers": word_speakers},
    )


def _utterance(index: int, item: Utterance) -> dict[str, object]:
    return {
        "utterance_index": index,
        "start_ms": item.start_ms,
        "end_ms": item.end_ms,
        "text": item.text,
        "confidence": round(item.confidence, 6),
    }


def _word(index: int, item: Word) -> dict[str, object]:
    return {
        "word_index": index,
        "start_ms": item.start_ms,
        "end_ms": item.end_ms,
        "text": item.text,
        "confidence": round(item.confidence, 6),
    }


def write_canonical_json(document: dict[str, object], path: Path) -> ArtifactMetadata:
    data = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(data)
    return ArtifactMetadata(len(data), hashlib.sha256(data).hexdigest())
