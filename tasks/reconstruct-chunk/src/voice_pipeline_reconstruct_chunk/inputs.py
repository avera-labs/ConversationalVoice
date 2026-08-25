from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from voice_pipeline_chunk_contracts import (
    parse_chunk_diarization,
    parse_separation_result,
    parse_transcription_artifact,
    parse_transcription_result,
)

from .artifacts import audio_identity, canonical_json_bytes
from .audio import read_wav_bytes, slice_wav_bytes
from .reference import longest_pure_interval, parse_reference_manifest


@dataclass(frozen=True, slots=True)
class UpstreamInputs:
    snapshot: object
    separation: object
    speaker_mapping: tuple[int, ...]
    transcript_identity: tuple[str, int, str]


@dataclass(frozen=True, slots=True)
class LoadedInputs:
    upstream: UpstreamInputs
    transcript: dict
    separated_audio: tuple[bytes, ...]
    references: tuple[dict, ...]
    reference_manifest_identity: tuple[str, int, str]


class InputLoader:
    def __init__(self, storage, policy):
        self.storage = storage
        self.policy = policy

    def validate(self, claim) -> UpstreamInputs:
        if (
            claim.lang not in {"en", "zh"}
            or not claim.chunk_audio_uri
            or not claim.audio_part_audio_uri
            or not claim.duration_ms
            or not isinstance(claim.diarizations, dict)
            or not isinstance(claim.separation, dict)
            or not isinstance(claim.transcription, dict)
            or not isinstance(claim.persona, dict)
            or not isinstance(claim.persona_result, dict)
        ):
            raise RuntimeError("invalid_reconstruction_input")
        snapshot = parse_chunk_diarization(
            claim.diarizations, duration_ms=claim.duration_ms
        )
        input_audio = claim.separation.get("input_audio")
        if not isinstance(input_audio, dict):
            raise TypeError("invalid_separation_input")
        separation = parse_separation_result(
            claim.separation,
            duration_ms=claim.duration_ms,
            speaker_ids=snapshot.speaker_ids,
            input_size_bytes=input_audio.get("size_bytes"),
            input_sha256=input_audio.get("sha256"),
            output_uris=self.storage.separation_uris(claim.chunk_audio_uri),
        )
        mapping = tuple(
            item.diarization_speaker_id for item in separation.speaker_audio
        )
        artifacts = claim.transcription.get("artifacts")
        if not isinstance(artifacts, dict):
            raise TypeError("invalid_transcription_input")
        metadata = tuple(
            (artifacts[name]["size_bytes"], artifacts[name]["sha256"])
            for name in ("transcript", "word_alignment")
        )
        transcription = parse_transcription_result(
            claim.transcription,
            speaker_audio=separation.speaker_audio,
            artifact_uris=self.storage.transcription_uris(claim.chunk_audio_uri),
            artifact_metadata=metadata,
            expected_language=claim.lang,
        )
        transcript = transcription["artifacts"]["transcript"]
        return UpstreamInputs(
            snapshot=snapshot,
            separation=separation,
            speaker_mapping=mapping,
            transcript_identity=(
                transcript["uri"],
                transcript["size_bytes"],
                transcript["sha256"],
            ),
        )

    def load(self, claim, workspace) -> LoadedInputs:
        upstream = self.validate(claim)
        transcript = self._download_transcript(
            claim,
            upstream.transcript_identity,
            upstream.speaker_mapping,
            workspace,
        )
        separated_audio = self._download_separated(
            claim, upstream.separation, workspace
        )
        references, manifest_identity = self._download_references(
            claim,
            upstream,
            separated_audio,
            workspace,
        )
        return LoadedInputs(
            upstream=upstream,
            transcript=transcript,
            separated_audio=separated_audio,
            references=references,
            reference_manifest_identity=manifest_identity,
        )

    def _download_transcript(self, claim, identity, mapping, workspace):
        self.storage.download(identity[0], workspace.transcript)
        payload = workspace.transcript.read_bytes()
        if (
            len(payload) != identity[1]
            or hashlib.sha256(payload).hexdigest() != identity[2]
        ):
            raise RuntimeError("input_transcript_identity_mismatch")
        document = parse_transcription_artifact(
            json.loads(payload),
            kind="transcript",
            duration_ms=claim.duration_ms,
            speaker_mapping=mapping,
            expected_language=claim.lang,
        )
        if canonical_json_bytes(document) != payload:
            raise RuntimeError("input_transcript_is_not_canonical")
        return document

    def _download_separated(self, claim, separation, workspace) -> tuple[bytes, ...]:
        payloads = []
        for speaker in separation.speaker_audio:
            path = workspace.separated(speaker.output_slot)
            self.storage.download(speaker.uri, path)
            payload = path.read_bytes()
            if (
                len(payload) != speaker.size_bytes
                or hashlib.sha256(payload).hexdigest() != speaker.sha256
            ):
                raise RuntimeError("separated_track_identity_mismatch")
            audio = read_wav_bytes(
                payload, expected_rate=self.policy.audio.input_sample_rate_hz
            )
            if audio.duration_ms != claim.duration_ms:
                raise RuntimeError("separated_track_duration_mismatch")
            payloads.append(payload)
        return tuple(payloads)

    def _download_references(self, claim, upstream, separated_audio, workspace):
        mapping = upstream.speaker_mapping
        separation = upstream.separation
        manifest_uri = self.storage.reference_manifest_uri(claim.audio_part_audio_uri)
        self.storage.download(manifest_uri, workspace.reference_manifest)
        manifest_bytes = workspace.reference_manifest.read_bytes()
        manifest_identity = (
            manifest_uri,
            len(manifest_bytes),
            hashlib.sha256(manifest_bytes).hexdigest(),
        )
        selected = parse_reference_manifest(
            json.loads(manifest_bytes), expected_speaker_ids=mapping
        )
        references = []
        for slot, (diarization_id, metadata, separated) in enumerate(
            zip(mapping, selected, separation.speaker_audio, strict=True)
        ):
            if metadata is not None:
                payload, source_audio, selection, expected_duration_ms = (
                    self._load_reference_sample(
                        claim,
                        workspace,
                        slot,
                        diarization_id,
                        metadata,
                    )
                )
                source = "diarization_reference"
            else:
                payload, source_audio, selection = self._slice_reference_fallback(
                    upstream,
                    separated_audio[slot],
                    separated,
                    diarization_id,
                )
                expected_duration_ms = None
                source = "separated_track_slice"
            audio = read_wav_bytes(
                payload, expected_rate=self.policy.audio.input_sample_rate_hz
            )
            if (
                expected_duration_ms is not None
                and audio.duration_ms != expected_duration_ms
            ):
                raise RuntimeError("speaker_reference_duration_mismatch")
            references.append(
                {
                    "speaker_id": slot,
                    "diarization_speaker_id": diarization_id,
                    "source": source,
                    "source_audio": source_audio,
                    "selection": selection,
                    "sample_audio": audio_identity(
                        payload,
                        duration_ms=audio.duration_ms,
                        sample_rate_hz=audio.sample_rate_hz,
                    ),
                    "bytes": payload,
                }
            )
        return tuple(references), manifest_identity

    def _load_reference_sample(self, claim, workspace, slot, diarization_id, metadata):
        uri = self.storage.reference_audio_uri(
            claim.audio_part_audio_uri, diarization_id
        )
        if metadata["uri"] != uri:
            raise RuntimeError("speaker_reference_uri_mismatch")
        self.storage.download(uri, workspace.sample(slot))
        payload = workspace.sample(slot).read_bytes()
        if (
            len(payload) != metadata["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != metadata["sha256"]
        ):
            raise RuntimeError("speaker_reference_identity_mismatch")
        return (
            payload,
            {
                "uri": uri,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            {"timebase": "audio_part", "segments": metadata["segments"]},
            metadata["total_duration_ms"],
        )

    def _slice_reference_fallback(
        self, upstream, separated_payload, separated, diarization_id
    ):
        start_ms, end_ms = longest_pure_interval(
            upstream.snapshot.segments,
            speaker_id=diarization_id,
            maximum_duration_ms=self.policy.reference.fallback_maximum_duration_ms,
            edge_trim_ms=self.policy.reference.fallback_edge_trim_ms,
        )
        payload = slice_wav_bytes(
            separated_payload,
            start_ms=start_ms,
            end_ms=end_ms,
            sample_rate_hz=self.policy.audio.input_sample_rate_hz,
        )
        return (
            payload,
            {
                "uri": separated.uri,
                "size_bytes": separated.size_bytes,
                "sha256": separated.sha256,
            },
            {
                "timebase": "chunk",
                "segments": [
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": end_ms - start_ms,
                    }
                ],
            },
        )
