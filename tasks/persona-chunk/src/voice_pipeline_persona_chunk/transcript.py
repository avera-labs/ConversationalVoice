from __future__ import annotations


def _srt_time(milliseconds: int) -> str:
    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    seconds = (milliseconds % 60_000) // 1_000
    remainder = milliseconds % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{remainder:03d}"


def transcript_to_srt(document: dict) -> str:
    """Build canonical SRT from an already validated transcript document."""
    utterances = []
    for speaker in document["speakers"]:
        slot = speaker["output_slot"]
        speaker_id = speaker["diarization_speaker_id"]
        for item in speaker["utterances"]:
            utterances.append(
                (
                    item["start_ms"],
                    item["end_ms"],
                    slot,
                    item["utterance_index"],
                    speaker_id,
                    item["text"],
                )
            )
    utterances.sort(key=lambda item: item[:4])
    lines: list[str] = []
    for index, (start, end, _slot, _utterance, speaker_id, text) in enumerate(
        utterances, 1
    ):
        lines.extend(
            [
                str(index),
                f"{_srt_time(start)} --> {_srt_time(end)}",
                f"[Speaker {speaker_id}]: {text}",
                "",
            ]
        )
    return "\n".join(lines)
