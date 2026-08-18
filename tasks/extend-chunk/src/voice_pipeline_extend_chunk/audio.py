from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactIdentity, file_identity


@dataclass(frozen=True, slots=True)
class WavAudio:
    frames: bytes
    sample_rate_hz: int
    frame_count: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class TrackArtifact:
    path: Path
    speaker_id: int
    diarization_speaker_id: int
    sample_rate_hz: int
    duration_ms: int
    identity: ArtifactIdentity


def read_wav_bytes(
    payload: bytes,
    *,
    expected_rate: int = 44100,
    maximum_duration_ms: int | None = 60000,
) -> WavAudio:
    try:
        with wave.open(io.BytesIO(payload), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != expected_rate
                or reader.getcomptype() != "NONE"
            ):
                raise ValueError("WAV format is invalid")
            frame_count = reader.getnframes()
            frames = reader.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError("payload is not a valid WAV") from exc
    if frame_count <= 0 or len(frames) != frame_count * 2:
        raise ValueError("WAV is empty or truncated")
    duration_ms = round(frame_count * 1000 / expected_rate)
    if duration_ms <= 0 or (
        maximum_duration_ms is not None and duration_ms > maximum_duration_ms
    ):
        raise ValueError("WAV duration is invalid")
    return WavAudio(frames, expected_rate, frame_count, duration_ms)


def slice_wav_bytes(
    payload: bytes, *, start_ms: int, end_ms: int, expected_rate: int = 16000
) -> tuple[bytes, WavAudio]:
    """Extract one exact millisecond range into a standalone PCM16 WAV."""

    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("WAV slice bounds are invalid")
    source = read_wav_bytes(
        payload, expected_rate=expected_rate, maximum_duration_ms=None
    )
    start_frame = round(start_ms * expected_rate / 1000)
    end_frame = round(end_ms * expected_rate / 1000)
    if end_frame > source.frame_count or end_frame <= start_frame:
        raise ValueError("WAV slice exceeds source bounds")
    frames = source.frames[start_frame * 2 : end_frame * 2]
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(expected_rate)
        writer.writeframes(frames)
    result = target.getvalue()
    return result, read_wav_bytes(result, expected_rate=expected_rate)


def pcm16_mono_to_wav(payload: bytes, *, sample_rate_hz: int) -> bytes:
    """Wrap raw little-endian mono PCM16 bytes in a WAV container."""

    if (
        not payload
        or len(payload) % 2
        or isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
    ):
        raise ValueError("PCM audio is invalid")
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(payload)
    return target.getvalue()


def write_wav(path: Path, frames: bytes, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(frames)


def assemble_tracks(
    script: dict,
    utterance_payloads: list[bytes],
    *,
    speaker_mapping: tuple[int, int],
    policy,
    track_paths: tuple[Path, Path],
) -> tuple[dict, tuple[TrackArtifact, TrackArtifact]]:
    if len(utterance_payloads) != len(script["utterances"]):
        raise ValueError("TTS output count is invalid")
    audio = [read_wav_bytes(payload) for payload in utterance_payloads]
    sample_rate_hz = 44100
    turn_gap_frames = round(policy.turn_gap_ms * sample_rate_hz / 1000)
    overlap_frames = round(policy.overlap_ms * sample_rate_hz / 1000)
    same_speaker_gap_frames = round(policy.same_speaker_gap_ms * sample_rate_hz / 1000)
    speaker_ends = [0, 0]
    start_frames: list[int] = []
    end_frames: list[int] = []
    global_end = 0
    for index, (utterance, generated) in enumerate(
        zip(script["utterances"], audio, strict=True)
    ):
        speaker_id = utterance["speaker_id"]
        if index == 0:
            start = 0
        elif utterance["placement"] == "sequential":
            start = global_end + turn_gap_frames
        else:
            previous_end = end_frames[-1]
            start = max(
                start_frames[-1],
                previous_end - overlap_frames,
                speaker_ends[speaker_id] + same_speaker_gap_frames,
            )
            if start >= previous_end:
                raise ValueError("requested overlap cannot be scheduled")
        end = start + generated.frame_count
        start_frames.append(start)
        end_frames.append(end)
        speaker_ends[speaker_id] = end
        global_end = max(global_end, end)

    track_frames = [bytearray(global_end * 2), bytearray(global_end * 2)]
    for utterance, generated, start in zip(
        script["utterances"], audio, start_frames, strict=True
    ):
        speaker_id = utterance["speaker_id"]
        offset = start * 2
        end = offset + len(generated.frames)
        if end > len(track_frames[speaker_id]):
            raise ValueError("rendered utterance exceeds track bounds")
        track_frames[speaker_id][offset:end] = generated.frames

    tracks: list[TrackArtifact] = []
    for speaker_id, path in enumerate(track_paths):
        write_wav(path, bytes(track_frames[speaker_id]), sample_rate_hz)
        tracks.append(
            TrackArtifact(
                path,
                speaker_id,
                speaker_mapping[speaker_id],
                sample_rate_hz,
                round(global_end * 1000 / sample_rate_hz),
                file_identity(path),
            )
        )
    transcript = {
        "schema_version": 1,
        "language": "en",
        "timebase": "dialogue_extension",
        "duration_ms": round(global_end * 1000 / sample_rate_hz),
        "speaker_mapping": script["speaker_mapping"],
        "utterances": [
            {
                **utterance,
                "start_ms": round(start * 1000 / sample_rate_hz),
                "end_ms": round(end * 1000 / sample_rate_hz),
            }
            for utterance, start, end in zip(
                script["utterances"], start_frames, end_frames, strict=True
            )
        ],
    }
    return transcript, (tracks[0], tracks[1])
