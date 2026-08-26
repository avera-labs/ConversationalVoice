import io
import wave

from voice_pipeline_chunk_contracts import AlignedTextUnit, build_segment_word_alignment

from voice_pipeline_extend_chunk.audio import assemble_tracks, slice_wav_bytes


def wav_bytes(duration_ms):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(44100)
        writer.writeframes(b"\1\0" * round(duration_ms * 44.1))
    return target.getvalue()


def script():
    base = {
        "instruction": "Speak naturally.",
    }
    return {
        "language": "en",
        "speaker_mapping": [
            {"speaker_id": 0, "diarization_speaker_id": 4},
            {"speaker_id": 1, "diarization_speaker_id": 7},
        ],
        "utterances": [
            {
                **base,
                "utterance_index": 0,
                "speaker_id": 0,
                "text": "A full sentence.",
                "text_with_audio_tags": "A full sentence.",
                "type": "dialogue",
                "placement": "sequential",
            },
            {
                **base,
                "utterance_index": 1,
                "speaker_id": 1,
                "text": "Yeah.",
                "text_with_audio_tags": "Yeah.",
                "type": "backchannel",
                "placement": "overlap_previous",
            },
            {
                **base,
                "utterance_index": 2,
                "speaker_id": 1,
                "text": "And then I understood.",
                "text_with_audio_tags": "And then I understood.",
                "type": "dialogue",
                "placement": "sequential",
            },
        ],
    }


def test_assembler_creates_equal_tracks_and_actual_transcript(tmp_path, policy):
    paths = (tmp_path / "speaker-0.wav", tmp_path / "speaker-1.wav")
    value = script()
    durations = [2000, 500, 1200]
    alignments = [
        build_segment_word_alignment(
            utterance["text_with_audio_tags"],
            [
                AlignedTextUnit(
                    "".join(
                        character
                        for character in utterance["text"]
                        if character.isalnum()
                    ),
                    100,
                    duration - 100,
                )
            ],
            duration_ms=duration,
        )
        for utterance, duration in zip(value["utterances"], durations, strict=True)
    ]
    transcript, tracks = assemble_tracks(
        value,
        [wav_bytes(duration) for duration in durations],
        alignments,
        speaker_mapping=(4, 7),
        policy=policy.timeline,
        track_paths=paths,
    )
    first, backchannel, final = transcript["utterances"]
    assert first["start_ms"] == 0
    assert backchannel["start_ms"] < first["end_ms"]
    assert final["start_ms"] > max(first["end_ms"], backchannel["end_ms"])
    assert first["word_alignment"][0]["start_ms"] == 100
    assert final["word_alignment"][0]["start_ms"] == final["start_ms"] + 100
    assert tracks[0].duration_ms == tracks[1].duration_ms == transcript["duration_ms"]
    with (
        wave.open(str(paths[0]), "rb") as left,
        wave.open(str(paths[1]), "rb") as right,
    ):
        assert left.getnframes() == right.getnframes()


def test_slice_wav_bytes_extracts_the_requested_16khz_range():
    source = io.BytesIO()
    with wave.open(source, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\1\0" * 16000)

    payload, audio = slice_wav_bytes(source.getvalue(), start_ms=250, end_ms=750)

    assert audio.sample_rate_hz == 16000
    assert audio.frame_count == 8000
    assert audio.duration_ms == 500
    with wave.open(io.BytesIO(payload), "rb") as reader:
        assert reader.getnframes() == 8000
