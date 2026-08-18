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
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_canonical_json(value: object, destination: Path) -> ArtifactIdentity:
    payload = canonical_json_bytes(value)
    destination.write_bytes(payload)
    return ArtifactIdentity(len(payload), hashlib.sha256(payload).hexdigest())


def file_identity(path: Path) -> ArtifactIdentity:
    payload = path.read_bytes()
    if not payload:
        raise ValueError("artifact is empty")
    return ArtifactIdentity(len(payload), hashlib.sha256(payload).hexdigest())
