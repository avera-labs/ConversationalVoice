import hashlib
import io
import json
import wave
from uuid import UUID

from voice_pipeline_chunk_contracts import (
    AlignedTextUnit,
    build_segment_word_alignment,
    parse_text_with_audio_tags,
)
from voice_pipeline_reconstruct_chunk.repository import Claim, Disposition
from voice_pipeline_reconstruct_chunk.storage import ObjectStorage
from voice_pipeline_reconstruct_chunk.task import Handler

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")
PART = UUID("22222222-2222-2222-2222-222222222222")
PART_BASE = "s3://bucket/raw/r/audio_parts/0"
CHUNK_BASE = f"{PART_BASE}/chunks/0"


def wav(rate, duration_ms, sample=b"\x01\x00"):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(sample * round(rate * duration_ms / 1000))
    return target.getvalue()


def identity(payload):
    return len(payload), hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


class Repo:
    def __init__(self, claim):
        self.value = claim

    def claim(self, _identifier):
        return self.value

    def complete(self, claim, result):
        self.completed = claim, result

    def fail(self, *args):
        self.failed = args

    def reject(self, *args):
        self.rejected = args

    def fail_publication(self, *args):
        self.publication_failed = args


class Storage:
    def __init__(self, objects):
        self.objects = objects
        self.paths = ObjectStorage(object(), "bucket")
        self.uploads = []

    def __getattr__(self, name):
        return getattr(self.paths, name)

    def download(self, uri, destination):
        destination.write_bytes(self.objects[uri])

    def upload_json(self, uri, source):
        self.uploads.append((uri, source.read_bytes()))

    def upload_wav(self, uri, source):
        self.uploads.append((uri, source.read_bytes()))


class Tags:
    def __init__(self):
        self.calls = []

    def analyze(self, audio, text):
        self.calls.append((audio, text))
        return {
            "text": text,
            "text_with_audio_tags": "[calm]" + text,
            "instruction": "Speak calmly and clearly.",
        }, {
            "model": "google/gemini-3.7-flash",
            "in_tokens": 1,
            "out_tokens": 1,
            "total_tokens": 2,
            "cost_usd": 0,
        }


class Tts:
    def __init__(self):
        self.calls = []

    def synthesize(self, utterance, reference_audio):
        self.calls.append((utterance, reference_audio))
        return wav(44100, 700 if len(self.calls) == 1 else 500)


class ForcedAlignment:
    def __init__(self):
        self.calls = []

    def align(self, audio, *, text_with_audio_tags, language):
        self.calls.append((audio, text_with_audio_tags, language))
        tagged = parse_text_with_audio_tags(text_with_audio_tags)
        duration_ms = 700 if len(self.calls) == 1 else 500
        unit = "".join(character for character in tagged.text if character.isalnum())
        return build_segment_word_alignment(
            text_with_audio_tags,
            [AlignedTextUnit(unit, 50, duration_ms - 50)],
            duration_ms=duration_ms,
        )


class Publisher:
    def publish(self, identifier):
        self.identifier = identifier
        return "task-id"


def build_claim_and_objects(language="en"):
    chinese = language == "zh"
    separated = (wav(16000, 3000), wav(16000, 3000, b"\x02\x00"))
    speaker_audio = []
    for slot, (speaker, payload) in enumerate(zip((4, 7), separated, strict=True)):
        size, sha = identity(payload)
        speaker_audio.append(
            {
                "output_slot": slot,
                "diarization_speaker_id": speaker,
                "uri": f"{CHUNK_BASE}/results/separated/speaker-{slot}.wav",
                "sample_rate_hz": 16000,
                "duration_ms": 3000,
                "size_bytes": size,
                "sha256": sha,
            }
        )
    separation = {
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
        "speaker_audio": speaker_audio,
        "audit": {
            "verdict": "ok",
            "reference_speaker_id": 4,
            "consistent_relation": "direct",
        },
    }
    transcript = {
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
                        "start_ms": 100,
                        "end_ms": 600,
                        "text": "你好。" if chinese else "Hello.",
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
                        "start_ms": 1600,
                        "end_ms": 2200,
                        "text": "好的。" if chinese else "Hi.",
                        "confidence": 0.8,
                    }
                ],
            },
        ],
    }
    transcript_payload = canonical(transcript)
    transcript_size, transcript_sha = identity(transcript_payload)
    transcription = {
        "schema_version": 1,
        "backend": transcript["backend"],
        "model": transcript["model"],
        "language": language,
        "input_speaker_audio": [
            {
                "output_slot": item["output_slot"],
                "diarization_speaker_id": item["diarization_speaker_id"],
                "uri": item["uri"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in speaker_audio
        ],
        "artifacts": {
            "transcript": {
                "uri": f"{CHUNK_BASE}/results/transcript.json",
                "size_bytes": transcript_size,
                "sha256": transcript_sha,
            },
            "word_alignment": {
                "uri": f"{CHUNK_BASE}/results/word_alignment.json",
                "size_bytes": 1,
                "sha256": "d" * 64,
            },
        },
    }
    samples = (wav(16000, 500, b"\x03\x00"), wav(16000, 500, b"\x04\x00"))
    manifest = {
        "schema_version": 1,
        "speakers": [
            {
                "speaker_id": speaker,
                "reference_audio": {
                    "uri": f"{PART_BASE}/speaker-references/speaker-{speaker}.wav",
                    "sample_rate_hz": 16000,
                    "size_bytes": identity(sample)[0],
                    "sha256": identity(sample)[1],
                    "segments": [{"start_ms": 0, "end_ms": 500, "duration_ms": 500}],
                    "effective_duration_ms": 500,
                    "total_duration_ms": 500,
                },
            }
            for speaker, sample in zip((4, 7), samples, strict=True)
        ],
    }
    objects = {
        f"{CHUNK_BASE}/results/transcript.json": transcript_payload,
        f"{CHUNK_BASE}/results/separated/speaker-0.wav": separated[0],
        f"{CHUNK_BASE}/results/separated/speaker-1.wav": separated[1],
        f"{PART_BASE}/speaker-references/references.json": json.dumps(
            manifest
        ).encode(),
        f"{PART_BASE}/speaker-references/speaker-4.wav": samples[0],
        f"{PART_BASE}/speaker-references/speaker-7.wav": samples[1],
    }
    claim = Claim(
        chunk_id=IDENTIFIER,
        disposition=Disposition.CLAIMED,
        status="reconstructing",
        audio_part_id=PART,
        chunk_audio_uri=f"{CHUNK_BASE}/audio.wav",
        audio_part_audio_uri=f"{PART_BASE}/audio.wav",
        lang=language,
        duration_ms=3000,
        diarizations={
            "schema_version": 1,
            "timebase": "chunk",
            "segments": [
                {"speaker": 4, "start_ms": 0, "end_ms": 1400, "duration_ms": 1400},
                {"speaker": 7, "start_ms": 1500, "end_ms": 2900, "duration_ms": 1400},
            ],
        },
        separation=separation,
        transcription=transcription,
        persona={"valid": True},
        persona_result={"valid": True},
    )
    return claim, objects


def test_handler_reconstructs_without_asr_and_publishes_extension(tmp_path, policy):
    claim, objects = build_claim_and_objects()
    repo = Repo(claim)
    storage = Storage(objects)
    tags = Tags()
    tts = Tts()
    publisher = Publisher()

    alignment = ForcedAlignment()
    outcome = Handler(
        repo,
        storage,
        tags,
        tts,
        publisher,
        policy,
        tmp_path,
        forced_aligner=alignment,
    )(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "reconstructed"
    assert outcome["utterance_count"] == 2
    assert publisher.identifier == IDENTIFIER
    assert len(tags.calls) == len(tts.calls) == 2
    assert len(alignment.calls) == 2
    assert all(call[1].startswith(b"RIFF") for call in tts.calls)
    result = repo.completed[1]
    assert result["models"] == {
        "audio_tags": "google/gemini-3.7-flash",
        "tts": "fish-audio/s2.1-pro",
        "forced_alignment": {
            "id": "Qwen/Qwen3-ForcedAligner-0.6B",
            "revision": policy.forced_alignment.revision,
        },
    }
    assert (
        result["artifacts"]["speaker_audio"][0]["duration_ms"]
        == result["actual_duration_ms"]
    )
    assert {uri.rsplit("/", 1)[-1] for uri, _payload in storage.uploads} == {
        "manifest.json",
        "transcript.json",
        "speaker-0.wav",
        "speaker-1.wav",
    }
    transcript_uri = f"{CHUNK_BASE}/results/reconstruction/transcript.json"
    first_utterance = json.loads(dict(storage.uploads)[transcript_uri])["utterances"][0]
    assert first_utterance["text"] == "Hello."
    assert first_utterance["text_with_audio_tags"] == "[calm]Hello."
    assert first_utterance["instruction"] == "Speak calmly and clearly."
    assert first_utterance["word_alignment"][0]["type"] == "audio_tag"
    assert first_utterance["word_alignment"][0]["start_ms"] == 150
    assert "tone" not in first_utterance
    assert "audio_tags" not in first_utterance
    assert not hasattr(repo, "failed")


def test_handler_accepts_tts_model_from_policy_without_capability_mapping(
    tmp_path, policy
):
    claim, objects = build_claim_and_objects()
    repo = Repo(claim)
    storage = Storage(objects)
    configured_model = "provider/plain-tts"
    configured_policy = policy.model_copy(
        update={"tts": policy.tts.model_copy(update={"model": configured_model})}
    )

    outcome = Handler(
        repo,
        storage,
        Tags(),
        Tts(),
        Publisher(),
        configured_policy,
        tmp_path,
        forced_aligner=ForcedAlignment(),
    )(str(IDENTIFIER))

    assert outcome["outcome"] == "reconstructed"
    assert repo.completed[1]["models"]["tts"] == configured_model
    assert not hasattr(repo, "failed")


def test_handler_reconstructs_chinese_and_publishes_extension(tmp_path, policy):
    claim, objects = build_claim_and_objects("zh")
    repo = Repo(claim)
    storage = Storage(objects)
    tags = Tags()
    tts = Tts()
    publisher = Publisher()

    outcome = Handler(
        repo,
        storage,
        tags,
        tts,
        publisher,
        policy,
        tmp_path,
        forced_aligner=ForcedAlignment(),
    )(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "reconstructed"
    assert publisher.identifier == IDENTIFIER
    assert [text for _audio, text in tags.calls] == ["你好。", "好的。"]
    assert repo.completed[1]["language"] == "zh"
    uploaded = dict(storage.uploads)
    transcript_uri = f"{CHUNK_BASE}/results/reconstruction/transcript.json"
    manifest_uri = f"{CHUNK_BASE}/results/reconstruction/manifest.json"
    assert json.loads(uploaded[transcript_uri])["language"] == "zh"
    assert json.loads(uploaded[manifest_uri])["language"] == "zh"
    assert not hasattr(repo, "failed")


def test_handler_passes_arbitrary_language_to_forced_aligner(tmp_path, policy):
    claim, objects = build_claim_and_objects("es")
    repo = Repo(claim)
    storage = Storage(objects)
    alignment = ForcedAlignment()

    outcome = Handler(
        repo,
        storage,
        Tags(),
        Tts(),
        Publisher(),
        policy,
        tmp_path,
        forced_aligner=alignment,
    )(str(IDENTIFIER))

    assert outcome["outcome"] == "reconstructed"
    assert [call[2] for call in alignment.calls] == ["es", "es"]
    assert repo.completed[1]["language"] == "es"
