from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from uuid import UUID

from celery import Celery
from voice_pipeline_chunk_contracts import (
    parse_chunk_diarization,
    parse_dialogue_extension_document,
    parse_dialogue_extension_transcript,
    parse_persona_document,
    parse_persona_result,
    parse_separation_result,
    parse_transcription_artifact,
    parse_transcription_result,
)
from voice_pipeline_task_contracts import EXTEND_CHUNK

from .artifacts import canonical_json_bytes, write_canonical_json
from .audio import assemble_tracks, read_wav_bytes, slice_wav_bytes
from .errors import safe_error
from .fish_audio import tts_text
from .reference import (
    SpeakerReferenceUnavailable,
    longest_pure_interval,
    parse_reference_manifest,
)
from .repository import Disposition
from .workspace import Workspace

logger = logging.getLogger(__name__)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Handler:
    def __init__(
        self,
        repository,
        storage,
        dialogue_client,
        fish_client,
        policy,
        workspace_parent=None,
    ):
        self.repository = repository
        self.storage = storage
        self.dialogue_client = dialogue_client
        self.fish_client = fish_client
        self.policy = policy
        self.workspace_parent = workspace_parent

    def __call__(self, value):
        try:
            identifier = UUID(value)
        except (AttributeError, TypeError, ValueError):
            self._finished(None, "invalid_chunk_id")
            raise
        try:
            claim = self.repository.claim(identifier)
        except Exception:
            self._finished(identifier, "claim_failed")
            raise
        if claim.disposition is Disposition.ALREADY_COMPLETED:
            self._validate_completed(claim)
        if claim.disposition is not Disposition.CLAIMED:
            self._finished(identifier, claim.disposition.value)
            return {"chunk_id": str(identifier), "outcome": claim.disposition.value}

        workspace = None
        try:
            workspace = Workspace(
                self.policy.task.workspace_prefix, self.workspace_parent
            )
            (
                snapshot,
                separation,
                mapping,
                transcript_identity,
                persona_identity,
            ) = self._validate_upstream(claim)
            transcript = self._download_transcript(
                claim, transcript_identity, mapping, workspace
            )
            references, manifest_identity = self._download_references(
                claim, snapshot, separation, mapping, workspace
            )
            reference_texts = [
                self.fish_client.transcribe_reference(reference["bytes"])
                for reference in references
            ]
            wire, usage = self.dialogue_client.extend(
                claim.persona, transcript, self.policy.dialogue
            )
            script = self._build_script(wire, usage, mapping)
            parse_dialogue_extension_document(
                script,
                speaker_mapping=mapping,
                model_id=self.policy.openrouter.model,
                target_duration_ms=self.policy.dialogue.target_duration_ms,
                config_version=self.policy.config_version,
                min_utterances=self.policy.dialogue.min_utterances,
                max_utterances=self.policy.dialogue.max_utterances,
            )
            script_metadata = write_canonical_json(script, workspace.script)

            synthesized = []
            for utterance in script["utterances"]:
                speaker_id = utterance["speaker_id"]
                synthesized.append(
                    self.fish_client.synthesize(
                        tts_text(utterance),
                        references[speaker_id]["bytes"],
                        reference_texts[speaker_id],
                    )
                )
            transcript_output, tracks = assemble_tracks(
                script,
                synthesized,
                speaker_mapping=mapping,
                policy=self.policy.timeline,
                track_paths=(workspace.track(0), workspace.track(1)),
            )
            parse_dialogue_extension_transcript(
                transcript_output, script=script, speaker_mapping=mapping
            )
            transcript_metadata = write_canonical_json(
                transcript_output, workspace.output_transcript
            )
            output_uris = self.storage.output_uris(claim.chunk_audio_uri)
            speaker_uris = output_uris["speaker_audio"]
            for track, uri in zip(tracks, speaker_uris, strict=True):
                self.storage.upload_wav(uri, track.path)
            self.storage.upload_json(
                output_uris["transcript"], workspace.output_transcript
            )
            self.storage.upload_json(output_uris["script"], workspace.script)

            result = self._build_result(
                claim,
                mapping,
                transcript_identity,
                persona_identity,
                manifest_identity,
                references,
                reference_texts,
                output_uris,
                script_metadata,
                transcript_metadata,
                transcript_output["duration_ms"],
                tracks,
            )
            self._validate_result(claim, result)
            self.repository.complete(claim, result)
            self._finished(identifier, "completed")
            return {
                "chunk_id": str(identifier),
                "outcome": "completed",
                "utterance_count": len(script["utterances"]),
                "duration_ms": transcript_output["duration_ms"],
            }
        except SpeakerReferenceUnavailable:
            self.repository.reject(
                identifier,
                safe_error(
                    "speaker_reference_unavailable",
                    self.policy.task.error_max_length,
                ),
            )
            self._finished(identifier, "rejected")
            return {"chunk_id": str(identifier), "outcome": "rejected"}
        except Exception:
            try:
                self.repository.fail(
                    identifier,
                    safe_error(
                        "dialogue_extension_failed", self.policy.task.error_max_length
                    ),
                )
            except Exception:  # noqa: BLE001 - preserve the original failure
                logger.error("extend_chunk.failure_persistence_failed")
            self._finished(identifier, "failed")
            raise
        finally:
            if workspace is not None:
                try:
                    workspace.close()
                except Exception:  # noqa: BLE001 - cleanup must not replace outcome
                    logger.error("extend_chunk.workspace_cleanup_failed")

    def _validate_upstream(self, claim):
        if (
            claim.lang != "en"
            or not claim.chunk_audio_uri
            or not claim.audio_part_audio_uri
            or not claim.duration_ms
            or not isinstance(claim.diarizations, dict)
            or not isinstance(claim.separation, dict)
            or not isinstance(claim.transcription, dict)
            or not isinstance(claim.persona, dict)
            or not isinstance(claim.persona_result, dict)
        ):
            raise RuntimeError("invalid_extension_input")
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
        )
        transcript_artifact = transcription["artifacts"]["transcript"]
        transcript_identity = self._identity_tuple(transcript_artifact, "transcript")
        persona_model = claim.persona_result.get("model")
        if not isinstance(persona_model, dict):
            raise TypeError("invalid_persona_input")
        persona_model_id = persona_model.get("id")
        parse_persona_document(
            claim.persona,
            speaker_mapping=mapping,
            model_id=persona_model_id,
            config_version="persona-v1",
        )
        persona_bytes = canonical_json_bytes(claim.persona)
        persona_artifact = claim.persona_result.get("artifact")
        if not isinstance(persona_artifact, dict):
            raise TypeError("invalid_persona_input")
        persona_identity = (
            self.storage.persona_uri(claim.chunk_audio_uri),
            len(persona_bytes),
            hashlib.sha256(persona_bytes).hexdigest(),
        )
        parse_persona_result(
            claim.persona_result,
            model_id=persona_model_id,
            config_version="persona-v1",
            input_audio=(
                claim.chunk_audio_uri,
                input_audio["size_bytes"],
                input_audio["sha256"],
            ),
            input_transcript=transcript_identity,
            artifact=persona_identity,
        )
        return snapshot, separation, mapping, transcript_identity, persona_identity

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
        )
        if canonical_json_bytes(document) != payload:
            raise RuntimeError("input_transcript_is_not_canonical")
        return document

    def _download_references(self, claim, snapshot, separation, mapping, workspace):
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
        for speaker_id, (diarization_id, metadata, separated) in enumerate(
            zip(mapping, selected, separation.speaker_audio, strict=True)
        ):
            path = workspace.reference(speaker_id)
            if metadata is not None:
                expected_uri = self.storage.reference_audio_uri(
                    claim.audio_part_audio_uri, diarization_id
                )
                if metadata["uri"] != expected_uri:
                    raise RuntimeError("speaker_reference_uri_mismatch")
                self.storage.download(expected_uri, path)
                payload = path.read_bytes()
                payload_sha256 = hashlib.sha256(payload).hexdigest()
                if (
                    len(payload) != metadata["size_bytes"]
                    or payload_sha256 != metadata["sha256"]
                ):
                    raise RuntimeError("speaker_reference_identity_mismatch")
                audio = read_wav_bytes(payload, expected_rate=16000)
                if audio.duration_ms != metadata["total_duration_ms"]:
                    raise RuntimeError("speaker_reference_duration_mismatch")
                source = "diarization_reference"
                source_audio = {
                    "uri": expected_uri,
                    "size_bytes": len(payload),
                    "sha256": payload_sha256,
                }
                selection = {
                    "timebase": "audio_part",
                    "segments": [
                        {
                            "start_ms": segment["start_ms"],
                            "end_ms": segment["end_ms"],
                            "duration_ms": segment["duration_ms"],
                        }
                        for segment in metadata["segments"]
                    ],
                }
            else:
                self.storage.download(separated.uri, path)
                separated_payload = path.read_bytes()
                separated_sha256 = hashlib.sha256(separated_payload).hexdigest()
                if (
                    len(separated_payload) != separated.size_bytes
                    or separated_sha256 != separated.sha256
                ):
                    raise RuntimeError("separated_track_identity_mismatch")
                separated_audio = read_wav_bytes(
                    separated_payload,
                    expected_rate=16000,
                    maximum_duration_ms=None,
                )
                if separated_audio.duration_ms != claim.duration_ms:
                    raise RuntimeError("separated_track_duration_mismatch")
                start_ms, end_ms = longest_pure_interval(
                    snapshot.segments, speaker_id=diarization_id
                )
                payload, audio = slice_wav_bytes(
                    separated_payload,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    expected_rate=16000,
                )
                payload_sha256 = hashlib.sha256(payload).hexdigest()
                source = "separated_track_slice"
                source_audio = {
                    "uri": separated.uri,
                    "size_bytes": len(separated_payload),
                    "sha256": separated_sha256,
                }
                selection = {
                    "timebase": "chunk",
                    "segments": [
                        {
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "duration_ms": end_ms - start_ms,
                        }
                    ],
                }
            references.append(
                {
                    "speaker_id": speaker_id,
                    "diarization_speaker_id": diarization_id,
                    "source": source,
                    "source_audio": source_audio,
                    "selection": selection,
                    "reference_audio": {
                        "sample_rate_hz": 16000,
                        "duration_ms": audio.duration_ms,
                        "size_bytes": len(payload),
                        "sha256": payload_sha256,
                    },
                    "bytes": payload,
                }
            )
        return references, manifest_identity

    def _build_script(self, wire, usage, mapping):
        if not isinstance(wire, Mapping) or set(wire) != {"utterances"}:
            raise TypeError("dialogue extension response fields are invalid")
        return {
            "schema_version": 1,
            "backend": "openrouter",
            "model": {
                "id": self.policy.openrouter.model,
                "config_version": self.policy.config_version,
            },
            "language": "en",
            "target_duration_ms": self.policy.dialogue.target_duration_ms,
            "speaker_mapping": [
                {"speaker_id": speaker_id, "diarization_speaker_id": diarization_id}
                for speaker_id, diarization_id in enumerate(mapping)
            ],
            "utterances": wire["utterances"],
            "usage": usage,
        }

    def _build_result(
        self,
        claim,
        mapping,
        transcript_identity,
        persona_identity,
        manifest_identity,
        references,
        reference_texts,
        output_uris,
        script_metadata,
        transcript_metadata,
        duration_ms,
        tracks,
    ):
        return {
            "schema_version": 1,
            "backend": "openrouter",
            "models": {
                "dialogue": {
                    "backend": "openrouter",
                    "id": self.policy.openrouter.model,
                    "config_version": self.policy.config_version,
                },
                "reference_asr": {
                    "backend": "openrouter",
                    "id": self.policy.fish_audio.transcription_model,
                    "config_version": self.policy.config_version,
                },
                "tts": {
                    "backend": "openrouter",
                    "id": self.policy.fish_audio.model,
                    "config_version": self.policy.config_version,
                },
            },
            "language": "en",
            "target_duration_ms": self.policy.dialogue.target_duration_ms,
            "actual_duration_ms": duration_ms,
            "inputs": {
                "transcript": self._identity(transcript_identity),
                "persona": self._identity(persona_identity),
                "speaker_reference_manifest": self._identity(manifest_identity),
                "speaker_references": [
                    {
                        "speaker_id": reference["speaker_id"],
                        "diarization_speaker_id": reference["diarization_speaker_id"],
                        "source": reference["source"],
                        "source_audio": reference["source_audio"],
                        "selection": reference["selection"],
                        "reference_audio": reference["reference_audio"],
                        "reference_text_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                    }
                    for reference, text in zip(references, reference_texts, strict=True)
                ],
            },
            "artifacts": {
                "script": {
                    "uri": output_uris["script"],
                    "size_bytes": script_metadata.size_bytes,
                    "sha256": script_metadata.sha256,
                },
                "transcript": {
                    "uri": output_uris["transcript"],
                    "size_bytes": transcript_metadata.size_bytes,
                    "sha256": transcript_metadata.sha256,
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

    def _validate_completed(self, claim):
        self._validate_upstream(claim)
        if not isinstance(claim.extension_result, dict):
            raise TypeError("invalid_completed_extension")
        self._validate_result(claim, claim.extension_result, current_policy=False)

    def _validate_result(self, claim, result, *, current_policy=True):
        root_fields = {
            "schema_version",
            "backend",
            "models",
            "language",
            "target_duration_ms",
            "actual_duration_ms",
            "inputs",
            "artifacts",
        }
        root = self._exact(result, root_fields, "extension result")
        if (
            root["schema_version"] != 1
            or root["backend"] != "openrouter"
            or root["language"] != "en"
            or not self._positive_integer(root["target_duration_ms"])
            or not self._positive_integer(root["actual_duration_ms"])
        ):
            raise ValueError("extension result identity is invalid")
        models = self._exact(
            root["models"], {"dialogue", "reference_asr", "tts"}, "models"
        )
        dialogue = self._exact(
            models["dialogue"], {"backend", "id", "config_version"}, "dialogue model"
        )
        reference_asr = self._exact(
            models["reference_asr"],
            {"backend", "id", "config_version"},
            "reference ASR model",
        )
        tts = self._exact(
            models["tts"],
            {"backend", "id", "config_version"},
            "TTS model",
        )
        if (
            dialogue["backend"] != "openrouter"
            or not self._model_id(dialogue["id"])
            or dialogue["config_version"] != "dialogue-extension-v1"
            or reference_asr["backend"] != "openrouter"
            or reference_asr["id"] != "fish-audio/transcribe-1"
            or reference_asr["config_version"] != "dialogue-extension-v1"
            or tts["backend"] != "openrouter"
            or tts["id"] != "fish-audio/s2.1-pro"
            or tts["config_version"] != "dialogue-extension-v1"
        ):
            raise ValueError("extension model identity is invalid")
        if current_policy and (
            root["target_duration_ms"] != self.policy.dialogue.target_duration_ms
            or dialogue["id"] != self.policy.openrouter.model
            or dialogue["config_version"] != self.policy.config_version
            or reference_asr["id"] != self.policy.fish_audio.transcription_model
            or reference_asr["config_version"] != self.policy.config_version
            or tts["id"] != self.policy.fish_audio.model
            or tts["config_version"] != self.policy.config_version
        ):
            raise ValueError("extension result disagrees with current policy")
        inputs = self._exact(
            root["inputs"],
            {
                "transcript",
                "persona",
                "speaker_reference_manifest",
                "speaker_references",
            },
            "inputs",
        )
        transcript = self._identity_tuple(inputs["transcript"], "transcript")
        persona = self._identity_tuple(inputs["persona"], "persona")
        manifest = self._identity_tuple(
            inputs["speaker_reference_manifest"], "speaker reference manifest"
        )
        transcript_artifact = claim.transcription["artifacts"]["transcript"]
        expected_transcript = (
            self.storage.transcription_uri(claim.chunk_audio_uri),
            transcript_artifact["size_bytes"],
            transcript_artifact["sha256"],
        )
        persona_bytes = canonical_json_bytes(claim.persona)
        expected_persona = (
            self.storage.persona_uri(claim.chunk_audio_uri),
            len(persona_bytes),
            hashlib.sha256(persona_bytes).hexdigest(),
        )
        if transcript != expected_transcript:
            raise ValueError("transcript identity is invalid")
        if persona != expected_persona:
            raise ValueError("persona identity is invalid")
        if manifest[0] != self.storage.reference_manifest_uri(
            claim.audio_part_audio_uri
        ):
            raise ValueError("speaker reference manifest URI is invalid")
        mapping = tuple(
            item["diarization_speaker_id"] for item in claim.separation["speaker_audio"]
        )
        snapshot = parse_chunk_diarization(
            claim.diarizations, duration_ms=claim.duration_ms
        )
        references = inputs["speaker_references"]
        if not isinstance(references, list) or len(references) != 2:
            raise ValueError("speaker references are invalid")
        for speaker_id, (raw, diarization_id) in enumerate(
            zip(references, mapping, strict=True)
        ):
            item = self._exact(
                raw,
                {
                    "speaker_id",
                    "diarization_speaker_id",
                    "source",
                    "source_audio",
                    "selection",
                    "reference_audio",
                    "reference_text_sha256",
                },
                "speaker reference",
            )
            if (
                item["speaker_id"] != speaker_id
                or item["diarization_speaker_id"] != diarization_id
                or item["source"]
                not in {"diarization_reference", "separated_track_slice"}
                or not isinstance(item["reference_text_sha256"], str)
                or not _SHA256.fullmatch(item["reference_text_sha256"])
            ):
                raise ValueError("speaker reference mapping is invalid")
            source_audio = self._identity_tuple(
                item["source_audio"], "speaker reference source audio"
            )
            selection = self._exact(
                item["selection"], {"timebase", "segments"}, "reference selection"
            )
            segments = selection["segments"]
            if not isinstance(segments, list) or not segments:
                raise ValueError("reference selection segments are invalid")
            parsed_segments = []
            for segment in segments:
                parsed = self._exact(
                    segment,
                    {"start_ms", "end_ms", "duration_ms"},
                    "reference selection segment",
                )
                if (
                    not self._nonnegative_integer(parsed["start_ms"])
                    or not self._positive_integer(parsed["end_ms"])
                    or not self._positive_integer(parsed["duration_ms"])
                    or parsed["end_ms"] <= parsed["start_ms"]
                    or parsed["duration_ms"] != parsed["end_ms"] - parsed["start_ms"]
                ):
                    raise ValueError("reference selection segment is invalid")
                parsed_segments.append(parsed)
            reference_audio = self._exact(
                item["reference_audio"],
                {"sample_rate_hz", "duration_ms", "size_bytes", "sha256"},
                "reference audio",
            )
            if (
                reference_audio["sample_rate_hz"] != 16000
                or not self._positive_integer(reference_audio["duration_ms"])
                or not self._positive_integer(reference_audio["size_bytes"])
                or not isinstance(reference_audio["sha256"], str)
                or not _SHA256.fullmatch(reference_audio["sha256"])
            ):
                raise ValueError("reference audio identity is invalid")
            if item["source"] == "diarization_reference":
                expected_uri = self.storage.reference_audio_uri(
                    claim.audio_part_audio_uri, diarization_id
                )
                expected_duration = sum(
                    segment["duration_ms"] for segment in parsed_segments
                ) + 500 * (len(parsed_segments) - 1)
                if (
                    selection["timebase"] != "audio_part"
                    or source_audio[0] != expected_uri
                    or source_audio[1:]
                    != (
                        reference_audio["size_bytes"],
                        reference_audio["sha256"],
                    )
                    or reference_audio["duration_ms"] != expected_duration
                ):
                    raise ValueError("diarization reference provenance is invalid")
            else:
                separated = claim.separation["speaker_audio"][speaker_id]
                start_ms, end_ms = longest_pure_interval(
                    snapshot.segments, speaker_id=diarization_id
                )
                if (
                    selection["timebase"] != "chunk"
                    or len(parsed_segments) != 1
                    or parsed_segments[0]
                    != {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": end_ms - start_ms,
                    }
                    or source_audio
                    != (
                        separated["uri"],
                        separated["size_bytes"],
                        separated["sha256"],
                    )
                    or reference_audio["duration_ms"] != end_ms - start_ms
                ):
                    raise ValueError("separated reference provenance is invalid")
        artifacts = self._exact(
            root["artifacts"], {"script", "transcript", "speaker_audio"}, "artifacts"
        )
        output_uris = self.storage.output_uris(claim.chunk_audio_uri)
        for name in ("script", "transcript"):
            identity = self._identity_tuple(artifacts[name], name)
            if identity[0] != output_uris[name]:
                raise ValueError("extension artifact URI is invalid")
        speaker_audio = artifacts["speaker_audio"]
        if not isinstance(speaker_audio, list) or len(speaker_audio) != 2:
            raise ValueError("extension speaker audio is invalid")
        for speaker_id, (raw, diarization_id, uri) in enumerate(
            zip(speaker_audio, mapping, output_uris["speaker_audio"], strict=True)
        ):
            item = self._exact(
                raw,
                {
                    "speaker_id",
                    "diarization_speaker_id",
                    "uri",
                    "sample_rate_hz",
                    "duration_ms",
                    "size_bytes",
                    "sha256",
                },
                "extension speaker audio",
            )
            self._identity_tuple(
                item,
                "extension speaker audio",
                extra={
                    "speaker_id",
                    "diarization_speaker_id",
                    "sample_rate_hz",
                    "duration_ms",
                },
            )
            if (
                item["speaker_id"] != speaker_id
                or item["diarization_speaker_id"] != diarization_id
                or item["uri"] != uri
                or item["sample_rate_hz"] != 44100
                or item["duration_ms"] != root["actual_duration_ms"]
            ):
                raise ValueError("extension speaker audio mapping is invalid")

    @staticmethod
    def _exact(value, fields, name):
        if not isinstance(value, Mapping) or set(value) != fields:
            raise TypeError(f"{name} fields are invalid")
        return value

    @classmethod
    def _identity_tuple(cls, value, name, extra=frozenset()):
        item = cls._exact(value, {"uri", "size_bytes", "sha256"} | set(extra), name)
        if (
            not isinstance(item["uri"], str)
            or not item["uri"]
            or not cls._positive_integer(item["size_bytes"])
            or not isinstance(item["sha256"], str)
            or not _SHA256.fullmatch(item["sha256"])
        ):
            raise ValueError(f"{name} identity is invalid")
        return item["uri"], item["size_bytes"], item["sha256"]

    @staticmethod
    def _positive_integer(value):
        return not isinstance(value, bool) and isinstance(value, int) and value > 0

    @staticmethod
    def _nonnegative_integer(value):
        return not isinstance(value, bool) and isinstance(value, int) and value >= 0

    @staticmethod
    def _canonical_string(value):
        return isinstance(value, str) and bool(value) and value == value.strip()

    @classmethod
    def _model_id(cls, value):
        return cls._canonical_string(value) and "/" in value

    @staticmethod
    def _identity(value):
        return {"uri": value[0], "size_bytes": value[1], "sha256": value[2]}

    @staticmethod
    def _finished(identifier, outcome):
        logger.info(
            "extend_chunk.finished",
            extra={
                "chunk_id": str(identifier) if identifier is not None else None,
                "outcome": outcome,
            },
        )


def register(app: Celery, handler):
    @app.task(
        name=EXTEND_CHUNK.name,
        queue=EXTEND_CHUNK.queue,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def extend_chunk(_task, value):
        return handler(value)

    return extend_chunk
