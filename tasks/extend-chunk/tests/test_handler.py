import hashlib
import io
import json
import wave
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from voice_pipeline_chunk_contracts import (
    AlignedTextUnit,
    build_segment_word_alignment,
    parse_text_with_audio_tags,
)

from voice_pipeline_extend_chunk.repository import Claim, Disposition
from voice_pipeline_extend_chunk.storage import ObjectStorage
from voice_pipeline_extend_chunk.task import Handler

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")
PART = UUID("22222222-2222-2222-2222-222222222222")
PART_BASE = "s3://bucket/raw_audios/r/audio_parts/0"
CHUNK_BASE = f"{PART_BASE}/chunks/0"


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def wav_bytes(sample_rate_hz, duration_ms):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(b"\1\0" * round(duration_ms * sample_rate_hz / 1000))
    return target.getvalue()


def identity(payload):
    return len(payload), hashlib.sha256(payload).hexdigest()


SEPARATED_PAYLOADS = (wav_bytes(16000, 3000), wav_bytes(16000, 3000))


def snapshot():
    return {
        "schema_version": 1,
        "timebase": "chunk",
        "segments": [
            {"speaker": 4, "start_ms": 0, "end_ms": 1500, "duration_ms": 1500},
            {"speaker": 7, "start_ms": 1500, "end_ms": 3000, "duration_ms": 1500},
        ],
    }


def separation():
    identities = tuple(identity(payload) for payload in SEPARATED_PAYLOADS)
    return {
        "schema_version": 1,
        "backend": "dialogue_sidon",
        "model": {
            "repo_id": "sarulab-speech/DialogueSidon",
            "revision": "a" * 40,
            "config_version": "sidon-v1",
            "inference_steps": 100,
        },
        "input_audio": {
            "sample_rate_hz": 16000,
            "duration_ms": 3000,
            "size_bytes": 100,
            "sha256": "b" * 64,
        },
        "speaker_audio": [
            {
                "output_slot": slot,
                "diarization_speaker_id": speaker,
                "uri": f"{CHUNK_BASE}/results/separated/speaker-{slot}.wav",
                "sample_rate_hz": 16000,
                "duration_ms": 3000,
                "size_bytes": identities[slot][0],
                "sha256": identities[slot][1],
            }
            for slot, speaker in enumerate((4, 7))
        ],
        "audit": {
            "verdict": "ok",
            "reference_speaker_id": 4,
            "consistent_relation": "direct",
        },
    }


def transcript(language="en"):
    chinese = language == "zh"
    return {
        "schema_version": 1,
        "backend": "paraformer_zh" if chinese else "parakeet_tdt",
        "model": {
            "repo_id": (
                "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                if chinese
                else "nvidia/parakeet-tdt-0.6b-v3"
            ),
            "revision": "v2.0.4" if chinese else "c" * 40,
            "config_version": "paraformer-zh-v1" if chinese else "parakeet-v1",
        },
        "language": language,
        "timebase": "chunk",
        "speakers": [
            {
                "output_slot": 0,
                "diarization_speaker_id": 4,
                "utterances": [
                    {
                        "utterance_index": 0,
                        "start_ms": 0,
                        "end_ms": 400,
                        "text": "我终于试过了。" if chinese else "I finally tried it.",
                        "confidence": 0.9,
                    }
                ],
            },
            {
                "output_slot": 1,
                "diarization_speaker_id": 7,
                "utterances": [
                    {
                        "utterance_index": 0,
                        "start_ms": 500,
                        "end_ms": 900,
                        "text": "感觉怎么样？" if chinese else "How was it?",
                        "confidence": 0.8,
                    }
                ],
            },
        ],
    }


def transcription_result(sep, transcript_payload, language="en"):
    chinese = language == "zh"
    size, sha256 = identity(transcript_payload)
    return {
        "schema_version": 1,
        "backend": "paraformer_zh" if chinese else "parakeet_tdt",
        "model": {
            "repo_id": (
                "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                if chinese
                else "nvidia/parakeet-tdt-0.6b-v3"
            ),
            "revision": "v2.0.4" if chinese else "c" * 40,
            "config_version": "paraformer-zh-v1" if chinese else "parakeet-v1",
        },
        "language": language,
        "input_speaker_audio": [
            {
                "output_slot": item["output_slot"],
                "diarization_speaker_id": item["diarization_speaker_id"],
                "uri": item["uri"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in sep["speaker_audio"]
        ],
        "artifacts": {
            "transcript": {
                "uri": f"{CHUNK_BASE}/results/transcript.json",
                "size_bytes": size,
                "sha256": sha256,
            },
            "word_alignment": {
                "uri": f"{CHUNK_BASE}/results/word_alignment.json",
                "size_bytes": 1,
                "sha256": "d" * 64,
            },
        },
    }


def persona(language="en"):
    chinese = language == "zh"
    base = {
        "name": None,
        "age": None,
        "ethnicity": None,
        "gender": None,
        "tag": "A natural conversational speaker.",
        "alpha": "low",
        "evidence": None,
        "primary_emotion": "neutral",
        "secondary_emotion": None,
        "emotion_intensity": "low",
        "laugh": False,
        "cry": False,
        "whisper": False,
        "shout": False,
        "sigh": False,
        "overall_tone": "calm",
    }
    return {
        "schema_version": 1,
        "backend": "openrouter",
        "config_version": "persona-v1",
        "language": language,
        "scene": {
            "description": (
                "两个人在交流一次体验。"
                if chinese
                else "Two people compare an experience."
            ),
            "overall_tone": "calm",
            "emotion_intensity": "low",
        },
        "speakers": [
            {"speaker_id": "4", **base},
            {"speaker_id": "7", **base},
        ],
        "usage": {
            "model": "xiaomi/mimo-v2.5",
            "in_tokens": 1,
            "out_tokens": 2,
            "total_tokens": 3,
            "cost_usd": 0.001,
        },
        "speaker_mapping": [
            {"output_slot": 0, "diarization_speaker_id": 4},
            {"output_slot": 1, "diarization_speaker_id": 7},
        ],
    }


def persona_result(document, transcript_payload, language="en"):
    persona_payload = canonical(document)
    transcript_size, transcript_sha = identity(transcript_payload)
    persona_size, persona_sha = identity(persona_payload)
    return {
        "schema_version": 1,
        "backend": "openrouter",
        "model": {"id": "xiaomi/mimo-v2.5", "config_version": "persona-v1"},
        "language": language,
        "input_audio": {
            "uri": f"{CHUNK_BASE}/audio.wav",
            "size_bytes": 100,
            "sha256": "b" * 64,
        },
        "input_transcript": {
            "uri": f"{CHUNK_BASE}/results/transcript.json",
            "size_bytes": transcript_size,
            "sha256": transcript_sha,
        },
        "artifact": {
            "uri": f"{CHUNK_BASE}/results/persona.json",
            "size_bytes": persona_size,
            "sha256": persona_sha,
        },
    }


def reference_manifest(reference_payloads):
    return {
        "schema_version": 1,
        "speakers": [
            {
                "speaker_id": speaker,
                "reference_audio": {
                    "uri": f"{PART_BASE}/speaker-references/speaker-{speaker}.wav",
                    "sample_rate_hz": 16000,
                    "size_bytes": identity(payload)[0],
                    "sha256": identity(payload)[1],
                    "segments": [
                        {"start_ms": 500, "end_ms": 5500, "duration_ms": 5000}
                    ],
                    "effective_duration_ms": 5000,
                    "total_duration_ms": 5000,
                },
            }
            for speaker, payload in zip((4, 7), reference_payloads, strict=True)
        ],
    }


class Repo:
    def __init__(self, transcript_payload, language="en"):
        sep = separation()
        document = persona(language)
        self.claim_value = Claim(
            IDENTIFIER,
            Disposition.CLAIMED,
            "extending",
            PART,
            f"{CHUNK_BASE}/audio.wav",
            f"{PART_BASE}/audio.wav",
            language,
            3000,
            snapshot(),
            sep,
            transcription_result(sep, transcript_payload, language),
            document,
            persona_result(document, transcript_payload, language),
            {"reconstruction": "valid"},
        )

    def claim(self, _identifier):
        return self.claim_value

    def complete(self, claim, result):
        self.completed = claim, result

    def reject(self, *args):
        self.rejected = args

    def fail(self, *args):
        self.failed = args


class Storage:
    def __init__(self, objects):
        self.objects = objects
        self.paths = ObjectStorage(object(), "bucket")
        self.uploads = []

    def __getattr__(self, name):
        return getattr(self.paths, name)

    def download(self, uri, destination):
        destination.write_bytes(self.objects[uri])

    def upload_wav(self, uri, source):
        self.uploads.append((uri, Path(source).read_bytes()))

    def upload_json(self, uri, source):
        self.uploads.append((uri, Path(source).read_bytes()))


class Dialogue:
    def extend(self, persona_document, transcript_document, _policy, language="en"):
        assert persona_document["language"] == language
        assert transcript_document["language"] == language
        assert persona_document["speaker_mapping"][0]["diarization_speaker_id"] == 4
        assert transcript_document["speakers"][1]["diarization_speaker_id"] == 7
        chinese = language == "zh"
        utterances = [
            {
                "utterance_index": 0,
                "speaker_id": 0,
                "text": "比我想象中好很多。"
                if chinese
                else "It was surprisingly good.",
                "text_with_audio_tags": (
                    "[pleased]比我想象中好很多。"
                    if chinese
                    else "[pleased]It was surprisingly good."
                ),
                "instruction": "愉快而自然地说。"
                if chinese
                else "Speak naturally with pleased surprise.",
                "type": "dialogue",
                "placement": "sequential",
            },
            {
                "utterance_index": 1,
                "speaker_id": 1,
                "text": "真的吗？" if chinese else "Really?",
                "text_with_audio_tags": "[curious]真的吗？"
                if chinese
                else "[curious]Really?",
                "instruction": "带着真诚的好奇快速回应。"
                if chinese
                else "Respond quickly with genuine curiosity.",
                "type": "backchannel",
                "placement": "overlap_previous",
            },
            {
                "utterance_index": 2,
                "speaker_id": 1,
                "text": "那我也想试试看。" if chinese else "Now I want to try it too.",
                "text_with_audio_tags": "那我也想试试看。"
                if chinese
                else "Now I want to try it too.",
                "instruction": "温和自然地继续。"
                if chinese
                else "Continue in a warm, natural voice.",
                "type": "dialogue",
                "placement": "sequential",
            },
        ]
        utterances.extend(
            {
                "utterance_index": index,
                "speaker_id": index % 2,
                "text": (
                    f"接下来的第{index}句话。"
                    if chinese
                    else f"Continuation line {index}."
                ),
                "text_with_audio_tags": (
                    f"接下来的第{index}句话。"
                    if chinese
                    else f"Continuation line {index}."
                ),
                "instruction": "自然地说。" if chinese else "Speak naturally.",
                "type": "dialogue",
                "placement": "sequential",
            }
            for index in range(3, 8)
        )
        return (
            {"utterances": utterances},
            {
                "model": "google/gemini-3.7-flash",
                "in_tokens": 10,
                "out_tokens": 20,
                "total_tokens": 30,
                "cost_usd": 0.01,
            },
        )


class Fish:
    def __init__(self):
        self.transcribed = []
        self.synthesized = []

    def transcribe_reference(self, payload, language="en"):
        self.transcribed.append((payload, language))
        return "参考语音。" if language == "zh" else "Reference words."

    def synthesize(self, utterance, reference_audio, reference_text):
        self.synthesized.append((utterance, reference_audio, reference_text))
        return wav_bytes(44100, 1000)


class ForcedAlignment:
    def __init__(self):
        self.calls = []

    def align(self, audio, *, text_with_audio_tags, language):
        self.calls.append((audio, text_with_audio_tags, language))
        tagged = parse_text_with_audio_tags(text_with_audio_tags)
        unit = "".join(character for character in tagged.text if character.isalnum())
        return build_segment_word_alignment(
            text_with_audio_tags,
            [AlignedTextUnit(unit, 100, 900)] if unit else [],
            duration_ms=1000,
        )


def make_handler(
    repository,
    storage,
    dialogue,
    fish,
    policy,
    workspace_parent,
    *,
    forced_aligner=None,
):
    return Handler(
        repository,
        storage,
        dialogue,
        fish,
        policy,
        workspace_parent,
        forced_aligner=forced_aligner or ForcedAlignment(),
    )


def build_objects(*, include_second_reference=True, language="en"):
    transcript_payload = canonical(transcript(language))
    references = (wav_bytes(16000, 5000), wav_bytes(16000, 5000))
    manifest = reference_manifest(references)
    if not include_second_reference:
        manifest["speakers"] = manifest["speakers"][:1]
    objects = {
        f"{CHUNK_BASE}/results/transcript.json": transcript_payload,
        f"{PART_BASE}/speaker-references/references.json": canonical(manifest),
        f"{PART_BASE}/speaker-references/speaker-4.wav": references[0],
        f"{PART_BASE}/speaker-references/speaker-7.wav": references[1],
        f"{CHUNK_BASE}/results/separated/speaker-0.wav": SEPARATED_PAYLOADS[0],
        f"{CHUNK_BASE}/results/separated/speaker-1.wav": SEPARATED_PAYLOADS[1],
    }
    return transcript_payload, objects


def test_handler_generates_extension_only_tracks_with_stable_mapping(tmp_path, policy):
    transcript_payload, objects = build_objects()
    repo = Repo(transcript_payload)
    storage = Storage(objects)
    fish = Fish()

    outcome = make_handler(repo, storage, Dialogue(), fish, policy, tmp_path)(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "completed"
    assert len(fish.transcribed) == 2
    assert len(fish.synthesized) == 8
    result = repo.completed[1]
    assert [
        (item["speaker_id"], item["diarization_speaker_id"])
        for item in result["artifacts"]["speaker_audio"]
    ] == [(0, 4), (1, 7)]
    assert [Path(uri).name for uri, _payload in storage.uploads[:2]] == [
        "speaker-0.wav",
        "speaker-1.wav",
    ]
    assert [
        reference["source"] for reference in result["inputs"]["speaker_references"]
    ] == ["diarization_reference", "diarization_reference"]
    assert all("audio.wav" not in uri for uri, _payload in storage.uploads)
    script_uri = f"{CHUNK_BASE}/results/dialogue-extension/script.json"
    first_utterance = json.loads(dict(storage.uploads)[script_uri])["utterances"][0]
    assert first_utterance["text"] == "It was surprisingly good."
    assert first_utterance["text_with_audio_tags"].startswith("[pleased]")
    assert first_utterance["instruction"]
    assert "tone" not in first_utterance
    assert "audio_tags" not in first_utterance
    transcript_uri = f"{CHUNK_BASE}/results/dialogue-extension/transcript.json"
    aligned = json.loads(dict(storage.uploads)[transcript_uri])["utterances"][0]
    assert [item["type"] for item in aligned["word_alignment"]] == [
        "audio_tag",
        "word",
    ]
    assert aligned["word_alignment"][0]["start_ms"] == 100
    assert (
        result["models"]["forced_alignment"]["id"]
        == "Qwen/Qwen3-ForcedAligner-0.6B"
    )
    assert not hasattr(repo, "failed")


def test_handler_accepts_tts_model_from_policy_without_capability_mapping(
    tmp_path, policy
):
    transcript_payload, objects = build_objects()
    repo = Repo(transcript_payload)
    storage = Storage(objects)
    configured_model = "provider/plain-tts"
    configured_policy = policy.model_copy(
        update={
            "fish_audio": policy.fish_audio.model_copy(
                update={"model": configured_model}
            )
        }
    )

    outcome = make_handler(
        repo, storage, Dialogue(), Fish(), configured_policy, tmp_path
    )(str(IDENTIFIER))

    assert outcome["outcome"] == "completed"
    assert repo.completed[1]["models"]["tts"]["id"] == configured_model
    assert not hasattr(repo, "failed")


def test_handler_generates_chinese_extension_and_preserves_language(tmp_path, policy):
    transcript_payload, objects = build_objects(language="zh")
    repo = Repo(transcript_payload, "zh")
    storage = Storage(objects)
    fish = Fish()

    outcome = make_handler(repo, storage, Dialogue(), fish, policy, tmp_path)(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "completed"
    assert [language for _payload, language in fish.transcribed] == ["zh", "zh"]
    assert fish.synthesized[0][0]["text"] == "比我想象中好很多。"
    assert (
        fish.synthesized[0][0]["text_with_audio_tags"] == "[pleased]比我想象中好很多。"
    )
    assert fish.synthesized[0][2] == "参考语音。"
    result = repo.completed[1]
    assert result["language"] == "zh"
    uploaded = dict(storage.uploads)
    script_uri = f"{CHUNK_BASE}/results/dialogue-extension/script.json"
    transcript_uri = f"{CHUNK_BASE}/results/dialogue-extension/transcript.json"
    assert json.loads(uploaded[script_uri])["language"] == "zh"
    assert json.loads(uploaded[transcript_uri])["language"] == "zh"
    assert "比我想象中好很多。".encode() in uploaded[script_uri]
    assert not hasattr(repo, "failed")

    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_COMPLETED,
        status="completed",
        extension_result=result,
    )
    storage.objects.clear()
    storage.uploads.clear()
    repeat_fish = Fish()
    repeat = make_handler(repo, storage, Dialogue(), repeat_fish, policy, tmp_path)(
        str(IDENTIFIER)
    )
    assert repeat["outcome"] == "already_completed"
    assert repeat_fish.transcribed == []
    assert storage.uploads == []


def test_missing_mapped_reference_uses_separated_track_fallback(tmp_path, policy):
    transcript_payload, objects = build_objects(include_second_reference=False)
    repo = Repo(transcript_payload)
    storage = Storage(objects)
    fish = Fish()

    outcome = make_handler(repo, storage, Dialogue(), fish, policy, tmp_path)(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "completed"
    assert len(fish.transcribed) == 2
    fallback = repo.completed[1]["inputs"]["speaker_references"][1]
    assert fallback["source"] == "separated_track_slice"
    assert fallback["source_audio"]["uri"].endswith("/results/separated/speaker-1.wav")
    assert fallback["selection"] == {
        "timebase": "chunk",
        "segments": [{"start_ms": 2000, "end_ms": 2500, "duration_ms": 500}],
    }
    assert fallback["reference_audio"]["duration_ms"] == 500
    assert not hasattr(repo, "failed")

    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_COMPLETED,
        status="completed",
        extension_result=repo.completed[1],
    )
    storage.objects.clear()
    repeat = make_handler(repo, storage, Dialogue(), Fish(), policy, tmp_path)(
        str(IDENTIFIER)
    )
    assert repeat["outcome"] == "already_completed"


def test_missing_reference_and_pure_interval_is_rejected_before_provider_calls(
    tmp_path, policy
):
    transcript_payload, objects = build_objects(include_second_reference=False)
    repo = Repo(transcript_payload)
    repo.claim_value = replace(
        repo.claim_value,
        diarizations={
            "schema_version": 1,
            "timebase": "chunk",
            "segments": [
                {"speaker": 4, "start_ms": 0, "end_ms": 3000, "duration_ms": 3000},
                {"speaker": 7, "start_ms": 500, "end_ms": 2500, "duration_ms": 2000},
            ],
        },
    )
    fish = Fish()

    outcome = make_handler(repo, Storage(objects), Dialogue(), fish, policy, tmp_path)(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "rejected"
    assert repo.rejected[0] == IDENTIFIER
    assert fish.transcribed == []
    assert not hasattr(repo, "failed")


def test_completed_invocation_validates_without_external_io(tmp_path, policy):
    transcript_payload, objects = build_objects()
    repo = Repo(transcript_payload)
    storage = Storage(objects)
    make_handler(repo, storage, Dialogue(), Fish(), policy, tmp_path)(str(IDENTIFIER))
    result = repo.completed[1]
    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_COMPLETED,
        status="completed",
        extension_result=result,
    )
    storage.objects.clear()
    storage.uploads.clear()
    fish = Fish()

    outcome = make_handler(repo, storage, Dialogue(), fish, policy, tmp_path)(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "already_completed"
    assert storage.uploads == []
    assert fish.transcribed == []


def test_completed_invocation_accepts_the_persisted_model_after_config_change(
    tmp_path, policy
):
    transcript_payload, objects = build_objects()
    repo = Repo(transcript_payload)
    storage = Storage(objects)
    make_handler(repo, storage, Dialogue(), Fish(), policy, tmp_path)(str(IDENTIFIER))
    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_COMPLETED,
        status="completed",
        extension_result=repo.completed[1],
    )
    changed_policy = policy.model_copy(
        update={
            "openrouter": policy.openrouter.model_copy(
                update={"model": "example/replacement"}
            ),
            "dialogue": policy.dialogue.model_copy(
                update={"target_duration_seconds": 180}
            ),
        }
    )

    outcome = make_handler(
        repo, storage, Dialogue(), Fish(), changed_policy, tmp_path
    )(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "already_completed"


def test_tts_failure_marks_claimed_row_failed(tmp_path, policy):
    transcript_payload, objects = build_objects()
    repo = Repo(transcript_payload)

    class FailedFish(Fish):
        def synthesize(self, *_args):
            raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        make_handler(
            repo, Storage(objects), Dialogue(), FailedFish(), policy, tmp_path
        )(
            str(IDENTIFIER)
        )
    assert repo.failed[0] == IDENTIFIER
