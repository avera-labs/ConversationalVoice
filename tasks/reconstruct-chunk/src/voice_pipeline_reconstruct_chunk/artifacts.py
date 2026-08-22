from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    size_bytes: int
    sha256: str


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def write_canonical_json(value: object, path: Path) -> ArtifactIdentity:
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    return ArtifactIdentity(len(payload), hashlib.sha256(payload).hexdigest())


def file_identity(path: Path) -> ArtifactIdentity:
    payload = path.read_bytes()
    return ArtifactIdentity(len(payload), hashlib.sha256(payload).hexdigest())


def audio_identity(payload: bytes, *, duration_ms: int, sample_rate_hz: int) -> dict:
    return {
        "sample_rate_hz": sample_rate_hz,
        "duration_ms": duration_ms,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
