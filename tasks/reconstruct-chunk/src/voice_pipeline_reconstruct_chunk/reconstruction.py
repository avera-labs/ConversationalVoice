from __future__ import annotations

from dataclasses import dataclass

from voice_pipeline_chunk_contracts import (
    offset_word_alignment,
    parse_reconstruction_transcript,
)

from .artifacts import audio_identity
from .audio import concatenate_reference, read_wav_bytes, slice_wav_bytes
from .timeline import schedule


@dataclass(frozen=True, slots=True)
class Reconstruction:
    transcript: dict
    generated_audio: list[bytes]
    segments: list[dict]
    audio_tag_usage: list[dict]


class Reconstructor:
    def __init__(self, tags_client, tts_client, forced_aligner, policy):
        self.tags_client = tags_client
        self.tts_client = tts_client
        self.forced_aligner = forced_aligner
        self.policy = policy

    def reconstruct(
        self,
        transcript,
        separated_audio,
        references,
        *,
        speaker_mapping,
        source_duration_ms,
        language="en",
    ) -> Reconstruction:
        source_utterances = flatten_utterances(transcript, speaker_mapping)
        generated_payloads = []
        generated_durations = []
        segment_records = []
        tag_usage = []
        segment_alignments = []
        for source in source_utterances:
            speaker_id = source["speaker_id"]
            separated_segment = slice_wav_bytes(
                separated_audio[speaker_id],
                start_ms=source["source_start_ms"],
                end_ms=source["source_end_ms"],
                sample_rate_hz=self.policy.audio.input_sample_rate_hz,
            )
            annotation, usage = self.tags_client.analyze(
                separated_segment, source["text"]
            )
            reference_segment = concatenate_reference(
                references[speaker_id]["bytes"],
                separated_segment,
                silence_ms=self.policy.audio.reference_silence_ms,
                sample_rate_hz=self.policy.audio.input_sample_rate_hz,
            )
            source.update(annotation)
            generated = self.tts_client.synthesize(source, reference_segment)
            generated_audio = read_wav_bytes(
                generated, expected_rate=self.policy.audio.output_sample_rate_hz
            )
            reference_audio = read_wav_bytes(
                reference_segment,
                expected_rate=self.policy.audio.input_sample_rate_hz,
            )
            generated_payloads.append(generated)
            generated_durations.append(generated_audio.duration_ms)
            segment_alignments.append(
                self.forced_aligner.align(
                    generated,
                    text_with_audio_tags=source["text_with_audio_tags"],
                    language=language,
                )
            )
            tag_usage.append(usage)
            segment_records.append(
                {
                    "utterance_index": source["utterance_index"],
                    "text_with_audio_tags": annotation["text_with_audio_tags"],
                    "instruction": annotation["instruction"],
                    "separated_segment": audio_identity(
                        separated_segment,
                        duration_ms=source["source_end_ms"] - source["source_start_ms"],
                        sample_rate_hz=self.policy.audio.input_sample_rate_hz,
                    ),
                    "reference_segment": audio_identity(
                        reference_segment,
                        duration_ms=reference_audio.duration_ms,
                        sample_rate_hz=self.policy.audio.input_sample_rate_hz,
                    ),
                    "reconstructed_segment": audio_identity(
                        generated,
                        duration_ms=generated_audio.duration_ms,
                        sample_rate_hz=self.policy.audio.output_sample_rate_hz,
                    ),
                }
            )

        scheduled = schedule(source_utterances, generated_durations)
        for utterance, alignment in zip(
            scheduled, segment_alignments, strict=True
        ):
            utterance["word_alignment"] = offset_word_alignment(
                alignment, utterance["start_ms"]
            )
        transcript_output = {
            "schema_version": 1,
            "language": language,
            "timebase": "reconstruction",
            "source_duration_ms": source_duration_ms,
            "duration_ms": max(item["end_ms"] for item in scheduled),
            "speaker_mapping": [
                {"speaker_id": slot, "diarization_speaker_id": diarization_id}
                for slot, diarization_id in enumerate(speaker_mapping)
            ],
            "utterances": scheduled,
        }
        parse_reconstruction_transcript(
            transcript_output,
            speaker_mapping=speaker_mapping,
            source_duration_ms=source_duration_ms,
            expected_language=language,
        )
        return Reconstruction(
            transcript=transcript_output,
            generated_audio=generated_payloads,
            segments=segment_records,
            audio_tag_usage=tag_usage,
        )


def flatten_utterances(transcript, speaker_mapping) -> list[dict]:
    result = []
    for speaker in transcript["speakers"]:
        slot = speaker["output_slot"]
        for item in speaker["utterances"]:
            result.append(
                {
                    "speaker_id": slot,
                    "diarization_speaker_id": speaker_mapping[slot],
                    "speaker_utterance_index": item["utterance_index"],
                    "text": item["text"],
                    "confidence": item["confidence"],
                    "source_start_ms": item["start_ms"],
                    "source_end_ms": item["end_ms"],
                }
            )
    result.sort(
        key=lambda item: (
            item["source_start_ms"],
            item["source_end_ms"],
            item["speaker_id"],
            item["speaker_utterance_index"],
        )
    )
    if not result:
        raise RuntimeError("reconstruction_utterances_unavailable")
    for index, item in enumerate(result):
        item["utterance_index"] = index
    return result
