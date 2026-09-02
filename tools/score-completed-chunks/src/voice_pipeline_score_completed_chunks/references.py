from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .audio import Audio, read_wav, slice_wav_payload
from .contracts import ReferenceDescriptor
from .errors import ScoringError
from .storage import ObjectStorage


@dataclass(frozen=True, slots=True)
class LoadedReference:
    descriptor: ReferenceDescriptor
    audio: Audio


def load_reference(
    storage: ObjectStorage, descriptor: ReferenceDescriptor
) -> LoadedReference:
    source = storage.download(descriptor.source_audio)
    if descriptor.source == "diarization_reference":
        payload = source
    else:
        payload = slice_wav_payload(
            source,
            segments=descriptor.selection,
            expected_rate=descriptor.sample_rate_hz,
        )
    if len(payload) != descriptor.size_bytes:
        raise ScoringError("reference_size_mismatch")
    if hashlib.sha256(payload).hexdigest() != descriptor.sha256:
        raise ScoringError("reference_sha256_mismatch")
    audio = read_wav(payload, expected_rate=descriptor.sample_rate_hz)
    if audio.duration_ms != descriptor.duration_ms:
        raise ScoringError("reference_duration_mismatch")
    return LoadedReference(descriptor, audio)
