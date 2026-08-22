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
    payload: bytes, *, expected_rate: int, maximum_duration_ms: int | None = None
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
    if maximum_duration_ms is not None and duration_ms > maximum_duration_ms:
        raise ValueError("WAV duration exceeds policy")
    return WavAudio(frames, expected_rate, frame_count, duration_ms)


def wav_bytes(frames: bytes, *, sample_rate_hz: int) -> bytes:
    if not frames or len(frames) % 2:
        raise ValueError("PCM16 frames are invalid")
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(frames)
    return target.getvalue()


def slice_wav_bytes(
    payload: bytes, *, start_ms: int, end_ms: int, sample_rate_hz: int = 16000
) -> bytes:
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("WAV slice bounds are invalid")
    source = read_wav_bytes(payload, expected_rate=sample_rate_hz)
    start_frame = round(start_ms * sample_rate_hz / 1000)
    end_frame = round(end_ms * sample_rate_hz / 1000)
    if end_frame > source.frame_count or end_frame <= start_frame:
        raise ValueError("WAV slice exceeds source bounds")
    return wav_bytes(
        source.frames[start_frame * 2 : end_frame * 2],
        sample_rate_hz=sample_rate_hz,
    )


def concatenate_reference(
    sample: bytes,
    utterance: bytes,
    *,
    silence_ms: int = 1000,
    sample_rate_hz: int = 16000,
) -> bytes:
    first = read_wav_bytes(sample, expected_rate=sample_rate_hz)
    second = read_wav_bytes(utterance, expected_rate=sample_rate_hz)
    silence = bytes(round(silence_ms * sample_rate_hz / 1000) * 2)
    return wav_bytes(
        first.frames + silence + second.frames,
        sample_rate_hz=sample_rate_hz,
    )


def pcm16_mono_to_wav(payload: bytes, *, sample_rate_hz: int) -> bytes:
    return wav_bytes(payload, sample_rate_hz=sample_rate_hz)


def render_tracks(
    utterances: list[dict],
    payloads: list[bytes],
    *,
    speaker_mapping: tuple[int, int],
    track_paths: tuple[Path, Path],
    sample_rate_hz: int = 44100,
) -> tuple[TrackArtifact, TrackArtifact]:
    if len(utterances) != len(payloads):
        raise ValueError("generated utterance count is invalid")
    decoded = [read_wav_bytes(item, expected_rate=sample_rate_hz) for item in payloads]
    duration_ms = max(item["end_ms"] for item in utterances)
    total_frames = max(
        round(duration_ms * sample_rate_hz / 1000),
        *(
            round(item["start_ms"] * sample_rate_hz / 1000) + audio.frame_count
            for item, audio in zip(utterances, decoded, strict=True)
        ),
    )
    tracks = [bytearray(total_frames * 2), bytearray(total_frames * 2)]
    for utterance, audio in zip(utterances, decoded, strict=True):
        start_frame = round(utterance["start_ms"] * sample_rate_hz / 1000)
        end_frame = start_frame + audio.frame_count
        if end_frame > total_frames:
            raise ValueError("generated segment exceeds reconstruction duration")
        offset = start_frame * 2
        tracks[utterance["speaker_id"]][offset : offset + len(audio.frames)] = (
            audio.frames
        )
    artifacts = []
    for speaker_id, path in enumerate(track_paths):
        path.write_bytes(
            wav_bytes(bytes(tracks[speaker_id]), sample_rate_hz=sample_rate_hz)
        )
        artifacts.append(
            TrackArtifact(
                path,
                speaker_id,
                speaker_mapping[speaker_id],
                sample_rate_hz,
                duration_ms,
                file_identity(path),
            )
        )
    return artifacts[0], artifacts[1]
