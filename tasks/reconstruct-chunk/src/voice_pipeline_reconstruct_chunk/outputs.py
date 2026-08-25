from __future__ import annotations

import re
from collections.abc import Mapping

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class OutputArtifacts:
    def __init__(self, storage, policy):
        self.storage = storage
        self.policy = policy

    def build_manifest(
        self,
        claim,
        loaded,
        reconstruction,
        output_uris,
        transcript_meta,
        tracks,
    ):
        mapping = loaded.upstream.speaker_mapping
        return {
            "schema_version": 1,
            "config_version": self.policy.config_version,
            "language": claim.lang,
            "models": {
                "audio_tags": {
                    "backend": "openrouter",
                    "id": self.policy.audio_tags.model,
                },
                "tts": {"backend": "openrouter", "id": self.policy.tts.model},
            },
            "speaker_mapping": _speaker_mapping(mapping),
            "inputs": {
                "transcript": stored_identity(loaded.upstream.transcript_identity),
                "separated_speaker_audio": [
                    {
                        "speaker_id": item["output_slot"],
                        "diarization_speaker_id": item["diarization_speaker_id"],
                        "uri": item["uri"],
                        "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in claim.separation["speaker_audio"]
                ],
                "speaker_reference_manifest": stored_identity(
                    loaded.reference_manifest_identity
                ),
                "speaker_references": [
                    _public_reference(item) for item in loaded.references
                ],
            },
            "policy": {
                "reference_silence_ms": self.policy.audio.reference_silence_ms,
                "timeline": "source-relative-v1",
            },
            "audio_tag_usage": reconstruction.audio_tag_usage,
            "segments": reconstruction.segments,
            "artifacts": {
                "transcript": {
                    "uri": output_uris["transcript"],
                    "size_bytes": transcript_meta.size_bytes,
                    "sha256": transcript_meta.sha256,
                },
                "speaker_audio": [
                    {
                        "speaker_id": track.speaker_id,
                        "uri": output_uris["speaker_audio"][track.speaker_id],
                        "size_bytes": track.identity.size_bytes,
                        "sha256": track.identity.sha256,
                    }
                    for track in tracks
                ],
            },
        }

    def build_result(
        self,
        claim,
        reconstruction,
        output_uris,
        manifest_meta,
        transcript_meta,
        tracks,
    ):
        mapping = reconstruction.transcript["speaker_mapping"]
        return {
            "schema_version": 1,
            "config_version": self.policy.config_version,
            "backend": "openrouter",
            "models": {
                "audio_tags": self.policy.audio_tags.model,
                "tts": self.policy.tts.model,
            },
            "language": claim.lang,
            "source_duration_ms": claim.duration_ms,
            "actual_duration_ms": reconstruction.transcript["duration_ms"],
            "utterance_count": len(reconstruction.transcript["utterances"]),
            "speaker_mapping": mapping,
            "artifacts": {
                "manifest": {
                    "uri": output_uris["manifest"],
                    "size_bytes": manifest_meta.size_bytes,
                    "sha256": manifest_meta.sha256,
                },
                "transcript": {
                    "uri": output_uris["transcript"],
                    "size_bytes": transcript_meta.size_bytes,
                    "sha256": transcript_meta.sha256,
                },
                "speaker_audio": [
                    {
                        "speaker_id": track.speaker_id,
                        "diarization_speaker_id": track.diarization_speaker_id,
                        "uri": output_uris["speaker_audio"][track.speaker_id],
                        "sample_rate_hz": track.sample_rate_hz,
                        "duration_ms": track.duration_ms,
                        "size_bytes": track.identity.size_bytes,
                        "sha256": track.identity.sha256,
                    }
                    for track in tracks
                ],
            },
        }

    def validate_result(self, claim, result, *, current_policy=True):
        required = {
            "schema_version",
            "config_version",
            "backend",
            "models",
            "language",
            "source_duration_ms",
            "actual_duration_ms",
            "utterance_count",
            "speaker_mapping",
            "artifacts",
        }
        if not isinstance(result, Mapping) or set(result) != required:
            raise TypeError("reconstruction result fields are invalid")
        if (
            result["schema_version"] != 1
            or result["config_version"] != "source-reconstruction-v1"
            or result["backend"] != "openrouter"
            or result["language"] != claim.lang
            or result["source_duration_ms"] != claim.duration_ms
            or not isinstance(result["actual_duration_ms"], int)
            or result["actual_duration_ms"] <= 0
            or not isinstance(result["utterance_count"], int)
            or result["utterance_count"] <= 0
        ):
            raise ValueError("reconstruction result identity is invalid")
        models = result["models"]
        if (
            not isinstance(models, Mapping)
            or set(models) != {"audio_tags", "tts"}
            or models["tts"] != "fish-audio/s2.1-pro"
        ):
            raise ValueError("reconstruction models are invalid")
        if current_policy and models != {
            "audio_tags": self.policy.audio_tags.model,
            "tts": self.policy.tts.model,
        }:
            raise ValueError("reconstruction models disagree with policy")
        mapping = tuple(
            item["diarization_speaker_id"] for item in claim.separation["speaker_audio"]
        )
        if result["speaker_mapping"] != _speaker_mapping(mapping):
            raise ValueError("reconstruction speaker mapping is invalid")
        artifacts = result["artifacts"]
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "manifest",
            "transcript",
            "speaker_audio",
        }:
            raise TypeError("reconstruction artifacts are invalid")
        output_uris = self.storage.output_uris(claim.chunk_audio_uri)
        for name in ("manifest", "transcript"):
            identity = validate_stored_identity(artifacts[name])
            if identity[0] != output_uris[name]:
                raise ValueError("reconstruction artifact URI is invalid")
        speaker_audio = artifacts["speaker_audio"]
        if not isinstance(speaker_audio, list) or len(speaker_audio) != 2:
            raise ValueError("reconstruction speaker audio is invalid")
        for slot, item in enumerate(speaker_audio):
            validate_stored_identity(
                item,
                extra={
                    "speaker_id",
                    "diarization_speaker_id",
                    "sample_rate_hz",
                    "duration_ms",
                },
            )
            if (
                item["speaker_id"] != slot
                or item["diarization_speaker_id"] != mapping[slot]
                or item["uri"] != output_uris["speaker_audio"][slot]
                or item["sample_rate_hz"] != 44100
                or item["duration_ms"] != result["actual_duration_ms"]
            ):
                raise ValueError("reconstruction speaker audio identity is invalid")


def stored_identity(value) -> dict:
    return {"uri": value[0], "size_bytes": value[1], "sha256": value[2]}


def validate_stored_identity(value, extra=frozenset()):
    fields = {"uri", "size_bytes", "sha256"} | set(extra)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TypeError("artifact identity fields are invalid")
    if (
        not isinstance(value["uri"], str)
        or not value["uri"]
        or isinstance(value["size_bytes"], bool)
        or not isinstance(value["size_bytes"], int)
        or value["size_bytes"] <= 0
        or not isinstance(value["sha256"], str)
        or not _SHA256.fullmatch(value["sha256"])
    ):
        raise ValueError("artifact identity is invalid")
    return value["uri"], value["size_bytes"], value["sha256"]


def _speaker_mapping(mapping) -> list[dict]:
    return [
        {"speaker_id": slot, "diarization_speaker_id": value}
        for slot, value in enumerate(mapping)
    ]


def _public_reference(item) -> dict:
    return {key: value for key, value in item.items() if key != "bytes"}
