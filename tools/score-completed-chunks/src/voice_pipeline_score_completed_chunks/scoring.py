from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .aggregation import build_group_row
from .audio import extract_active_audio, read_wav
from .contracts import GROUPS, parse_group, parse_references, validate_transcript
from .dnsmos import DnsmosScorer
from .errors import ScoringError, error_code
from .nisqa import NisqaScorer
from .references import load_reference
from .repository import CompletedChunk
from .speaker_similarity import SpeakerSimilarityScorer
from .storage import ObjectStorage


@dataclass(slots=True)
class ScoreEngine:
    storage: ObjectStorage
    nisqa: NisqaScorer
    dnsmos: DnsmosScorer
    speaker: SpeakerSimilarityScorer
    model_fingerprint: str

    def _failure_row(
        self,
        chunk: CompletedChunk,
        group: str,
        speaker_id: int,
        error: BaseException,
    ) -> dict:
        return {
            "schema_version": 1,
            "chunk_id": str(chunk.chunk_id),
            "language": chunk.language,
            "group": group,
            "speaker_id": speaker_id,
            "status": "failed",
            "error_code": error_code(error),
            "model_fingerprint": self.model_fingerprint,
        }

    def _resume_key(self, track_sha256: str, reference_sha256: str) -> str:
        value = f"{track_sha256}:{reference_sha256}:{self.model_fingerprint}"
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    def score_chunk(
        self,
        chunk: CompletedChunk,
        *,
        existing: dict[tuple[object, object, object], dict],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        speaker_rows: list[dict] = []
        group_rows: list[dict] = []
        failures: list[dict] = []
        try:
            reference_descriptors = parse_references(chunk.final_results)
            loaded_references = tuple(
                load_reference(self.storage, descriptor)
                for descriptor in reference_descriptors
            )
            reference_embeddings = tuple(
                self.speaker.embedding(reference.audio)
                for reference in loaded_references
            )
        except Exception as exc:
            for group in GROUPS:
                rows = [
                    self._failure_row(chunk, group, speaker_id, exc)
                    for speaker_id in range(2)
                ]
                speaker_rows.extend(rows)
                group_rows.append(
                    build_group_row(
                        chunk_id=str(chunk.chunk_id),
                        language=chunk.language,
                        group=group,
                        speaker_rows=rows,
                    )
                )
            failures.append(
                {
                    "chunk_id": str(chunk.chunk_id),
                    "scope": "references",
                    "error_code": error_code(exc),
                }
            )
            return speaker_rows, group_rows, failures

        for group_name in GROUPS:
            current_rows: list[dict] = []
            try:
                group = parse_group(chunk.final_results, group_name)
                if group.language != chunk.language:
                    raise ScoringError("chunk_language_mismatch")
                for track, reference in zip(
                    group.tracks, loaded_references, strict=True
                ):
                    if (
                        track.diarization_speaker_id
                        != reference.descriptor.diarization_speaker_id
                    ):
                        raise ScoringError("reference_mapping_mismatch")
                transcript_payload = self.storage.download(group.transcript)
                try:
                    transcript_json = json.loads(transcript_payload)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ScoringError("invalid_transcript_json") from exc
                transcript = validate_transcript(transcript_json, group=group)
                for track in group.tracks:
                    resume_key = self._resume_key(
                        track.artifact.sha256,
                        loaded_references[track.speaker_id].descriptor.sha256,
                    )
                    previous = existing.get(
                        (str(chunk.chunk_id), group_name, track.speaker_id)
                    )
                    if (
                        previous
                        and previous.get("status") == "success"
                        and previous.get("resume_key") == resume_key
                    ):
                        current_rows.append(previous)
                        continue
                    try:
                        payload = self.storage.download(track.artifact)
                        audio = read_wav(payload, expected_rate=track.sample_rate_hz)
                        if audio.duration_ms != track.duration_ms:
                            raise ScoringError("track_duration_mismatch")
                        active = extract_active_audio(
                            audio, transcript, speaker_id=track.speaker_id
                        )
                        output_embedding = self.speaker.embedding(
                            (active.samples, active.sample_rate_hz)
                        )
                        similarities = [
                            self.speaker.similarity(output_embedding, embedding)
                            for embedding in reference_embeddings
                        ]
                        matched = similarities[track.speaker_id]
                        other = similarities[1 - track.speaker_id]
                        row = {
                            "schema_version": 1,
                            "chunk_id": str(chunk.chunk_id),
                            "language": chunk.language,
                            "group": group_name,
                            "speaker_id": track.speaker_id,
                            "diarization_speaker_id": track.diarization_speaker_id,
                            "track_uri": track.artifact.uri,
                            "track_sha256": track.artifact.sha256,
                            "reference_source": loaded_references[
                                track.speaker_id
                            ].descriptor.source,
                            "reference_sha256": loaded_references[
                                track.speaker_id
                            ].descriptor.sha256,
                            "track_duration_ms": track.duration_ms,
                            "active_speech_duration_ms": active.active_duration_ms,
                            "active_ratio": active.active_duration_ms
                            / track.duration_ms,
                            "active_interval_count": active.interval_count,
                            **self.nisqa.score(active.samples, active.sample_rate_hz),
                            **self.dnsmos.score(active.samples, active.sample_rate_hz),
                            "speaker_similarity": matched,
                            "other_speaker_similarity": other,
                            "same_speaker_margin": matched - other,
                            "similarity_matrix_row": similarities,
                            "status": "success",
                            "error_code": None,
                            "model_fingerprint": self.model_fingerprint,
                            "resume_key": resume_key,
                        }
                    except Exception as exc:
                        row = self._failure_row(
                            chunk, group_name, track.speaker_id, exc
                        )
                        row.update(
                            {
                                "diarization_speaker_id": track.diarization_speaker_id,
                                "track_uri": track.artifact.uri,
                                "track_sha256": track.artifact.sha256,
                                "reference_sha256": loaded_references[
                                    track.speaker_id
                                ].descriptor.sha256,
                                "resume_key": resume_key,
                            }
                        )
                        failures.append(
                            {
                                "chunk_id": str(chunk.chunk_id),
                                "scope": f"{group_name}:speaker-{track.speaker_id}",
                                "error_code": error_code(exc),
                            }
                        )
                    current_rows.append(row)
            except Exception as exc:
                current_rows = [
                    self._failure_row(chunk, group_name, speaker_id, exc)
                    for speaker_id in range(2)
                ]
                failures.append(
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "scope": group_name,
                        "error_code": error_code(exc),
                    }
                )
            speaker_rows.extend(current_rows)
            group_rows.append(
                build_group_row(
                    chunk_id=str(chunk.chunk_id),
                    language=chunk.language,
                    group=group_name,
                    speaker_rows=current_rows,
                )
            )
        return speaker_rows, group_rows, failures
