from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import __version__
from .asr import OpenRouterAsrClient, error_rate
from .audio import mix_mono_tracks, read_wav, slice_audio, wav_bytes_from_samples
from .audio_tag_accuracy import AudioTagScoreEngine
from .contracts import GROUPS, parse_group, validate_transcript
from .errors import error_code
from .interaction_config import InteractionConfig
from .interaction_scoring import InteractionScoreEngine
from .nonverbal import DisabledNonverbalDetector
from .reporting import build_score_report, render_artifacts
from .repository import CompletedChunk
from .scoring import ScoreEngine
from .vad import EnergyVad

_AUDIO_TAG = re.compile(r"\[[^\[\]\r\n]+\]")


def scoring_code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ChunkScoreService:
    """Run one completed chunk evaluation without changing pipeline status."""

    def __init__(
        self,
        *,
        storage,
        nisqa,
        dnsmos,
        speaker,
        asr: OpenRouterAsrClient,
        audio_tag: AudioTagScoreEngine,
    ):
        self.storage = storage
        self.nisqa = nisqa
        self.dnsmos = dnsmos
        self.speaker = speaker
        self.asr = asr
        self.audio_tag = audio_tag
        interaction_config = InteractionConfig()
        self.interaction = InteractionScoreEngine(
            storage=storage,
            config=interaction_config,
            vad=EnergyVad(interaction_config),
            nonverbal=DisabledNonverbalDetector(),
        )
        self.code_fingerprint = scoring_code_fingerprint()
        fingerprint_source = json.dumps(
            {
                "scoring_code_sha256": self.code_fingerprint,
                "nisqa": nisqa.manifest(),
                "dnsmos": dnsmos.manifest(),
                "speaker": speaker.manifest(),
                "asr": asr.manifest(),
                "audio_tag": audio_tag.evaluator.manifest(),
                "interaction": self.interaction.manifest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.model_fingerprint = hashlib.sha256(fingerprint_source).hexdigest()
        self.acoustic = ScoreEngine(
            storage, nisqa, dnsmos, speaker, self.model_fingerprint
        )

    def score(self, chunk: CompletedChunk) -> tuple[dict, dict[str, bytes]]:
        speaker_rows, group_rows, failures = self.acoustic.score_chunk(
            chunk, existing={}
        )
        event_rows, interaction_rows, _declared_rows, interaction_failures = (
            self.interaction.score_chunk(chunk)
        )
        failures.extend(interaction_failures)
        audio_tag_rows, audio_tag_failures = self.audio_tag.score_chunk(
            chunk, existing={}
        )
        failures.extend(audio_tag_failures)
        asr_rows, asr_failures = self._score_asr(chunk)
        failures.extend(asr_failures)
        manifest = {
            "schema_version": 2,
            "tool_version": __version__,
            "chunk_id": str(chunk.chunk_id),
            "source_updated_at": chunk.updated_at.isoformat(),
            "model_fingerprint": self.model_fingerprint,
            "scoring_code_sha256": self.code_fingerprint,
            "device": "cpu",
            "models": {
                "nisqa": self.nisqa.manifest(),
                "dnsmos": self.dnsmos.manifest(),
                "speaker_similarity": self.speaker.manifest(),
                "asr": self.asr.manifest(),
                "audio_tag": self.audio_tag.evaluator.manifest(),
                "interaction": self.interaction.manifest(),
            },
        }
        report = build_score_report(
            chunk_id=str(chunk.chunk_id),
            language=chunk.language,
            group_rows=group_rows,
            speaker_rows=speaker_rows,
            interaction_rows=interaction_rows,
            asr_rows=asr_rows,
            audio_tag_rows=audio_tag_rows,
            failures=failures,
            manifest=manifest,
        )
        return report, render_artifacts(report, event_rows, audio_tag_rows)

    def _score_asr(self, chunk: CompletedChunk) -> tuple[list[dict], list[dict]]:
        rows: list[dict] = []
        failures: list[dict] = []
        for group_name in GROUPS:
            try:
                group = parse_group(chunk.final_results, group_name)
                transcript = validate_transcript(
                    json.loads(self.storage.download(group.transcript)), group=group
                )
                utterances = sorted(
                    transcript["utterances"],
                    key=lambda value: (
                        value.get("start_ms", 0),
                        value.get("end_ms", 0),
                        value.get("speaker_id", 0),
                    ),
                )
                reference = " ".join(
                    _AUDIO_TAG.sub("", str(value.get("text", ""))).strip()
                    for value in utterances
                    if str(value.get("text", "")).strip()
                )
                tracks = tuple(
                    read_wav(
                        self.storage.download(track.artifact),
                        expected_rate=track.sample_rate_hz,
                    )
                    for track in group.tracks
                )
                audio = mix_mono_tracks(tracks[0], tracks[1])
                asr = self._transcribe_windows(
                    audio,
                    utterances=utterances,
                    language=chunk.language,
                )
                rows.append(
                    {
                        "schema_version": 1,
                        "chunk_id": str(chunk.chunk_id),
                        "language": chunk.language,
                        "group": group_name,
                        "status": "success",
                        "model": self.asr.model,
                        "generation_ids": asr["generation_ids"],
                        "reference": reference,
                        "hypothesis": asr["hypothesis"],
                        "error": error_rate(
                            reference, asr["hypothesis"], language=chunk.language
                        ),
                        "segments": asr["segments"],
                        "usage": asr["usage"],
                    }
                )
            except Exception as exc:
                code = error_code(exc)
                rows.append(
                    {
                        "schema_version": 1,
                        "chunk_id": str(chunk.chunk_id),
                        "language": chunk.language,
                        "group": group_name,
                        "status": "failed",
                        "error_code": code,
                    }
                )
                failures.append(
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "scope": f"asr:{group_name}",
                        "error_code": code,
                    }
                )
        return rows, failures

    def _transcribe_windows(
        self, audio: bytes, *, utterances: list[dict], language: str
    ) -> dict[str, object]:
        """Split long timelines at utterance ends to avoid provider timeouts."""

        source = read_wav(audio, expected_rate=16_000)
        maximum_window_ms = 50_000
        utterance_ends = sorted(
            {
                min(source.duration_ms, int(value.get("end_ms", 0)))
                for value in utterances
                if isinstance(value.get("end_ms"), int)
                and 0 < int(value["end_ms"]) < source.duration_ms
            }
        )
        boundaries = [0]
        while boundaries[-1] + maximum_window_ms < source.duration_ms:
            candidates = [
                value
                for value in utterance_ends
                if boundaries[-1] + 1_000 <= value <= boundaries[-1] + maximum_window_ms
            ]
            boundaries.append(
                max(candidates) if candidates else boundaries[-1] + maximum_window_ms
            )
        boundaries.append(source.duration_ms)
        segments: list[dict[str, object]] = []
        usage: dict[str, float] = {}
        hypotheses: list[str] = []
        generation_ids: list[str] = []
        for start_ms, end_ms in zip(boundaries, boundaries[1:]):
            clip = slice_audio(source, start_ms=start_ms, end_ms=end_ms)
            payload = wav_bytes_from_samples(
                clip.samples, sample_rate_hz=clip.sample_rate_hz
            )
            result = self.asr.transcribe(payload, language=language)
            hypotheses.append(result.text)
            if result.generation_id:
                generation_ids.append(result.generation_id)
            for name, value in result.usage.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    usage[name] = usage.get(name, 0.0) + float(value)
            segments.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "hypothesis": result.text,
                    "generation_id": result.generation_id,
                    "usage": result.usage,
                }
            )
        return {
            "hypothesis": " ".join(value for value in hypotheses if value).strip(),
            "generation_ids": generation_ids,
            "segments": segments,
            "usage": usage,
        }
