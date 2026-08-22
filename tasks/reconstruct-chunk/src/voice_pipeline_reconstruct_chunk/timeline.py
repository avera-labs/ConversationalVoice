from __future__ import annotations


def schedule(utterances: list[dict], durations_ms: list[int]) -> list[dict]:
    """Adapt source gaps and relative overlap onsets to generated durations."""

    if not utterances or len(utterances) != len(durations_ms):
        raise ValueError("timeline inputs are invalid")
    result: list[dict] = []
    speaker_ends = [0, 0]
    previous_start = 0
    for index, (source, duration_ms) in enumerate(
        zip(utterances, durations_ms, strict=True)
    ):
        if duration_ms <= 0:
            raise ValueError("generated duration is invalid")
        if index == 0:
            start = source["source_start_ms"]
            relation = "leading"
            anchor_index = None
        else:
            active = [
                item
                for item in result
                if item["source_start_ms"]
                <= source["source_start_ms"]
                < item["source_end_ms"]
            ]
            if active:
                anchor = max(
                    active,
                    key=lambda item: (item["source_start_ms"], item["utterance_index"]),
                )
                anchor_index = anchor["utterance_index"]
                if source["source_start_ms"] == anchor["source_start_ms"]:
                    relation = "simultaneous"
                    start = anchor["start_ms"]
                else:
                    relation = "overlap"
                    source_duration = (
                        anchor["source_end_ms"] - anchor["source_start_ms"]
                    )
                    ratio = (
                        source["source_start_ms"] - anchor["source_start_ms"]
                    ) / source_duration
                    generated_anchor_duration = anchor["end_ms"] - anchor["start_ms"]
                    start = anchor["start_ms"] + round(
                        ratio * generated_anchor_duration
                    )
            else:
                ended = [
                    item
                    for item in result
                    if item["source_end_ms"] <= source["source_start_ms"]
                ]
                anchor = max(
                    ended,
                    key=lambda item: (item["source_end_ms"], item["utterance_index"]),
                )
                anchor_index = anchor["utterance_index"]
                relation = "gap"
                gap_ms = source["source_start_ms"] - anchor["source_end_ms"]
                start = anchor["end_ms"] + gap_ms
            start = max(start, previous_start, speaker_ends[source["speaker_id"]])
        end = start + duration_ms
        scheduled = {
            **source,
            "start_ms": start,
            "end_ms": end,
            "relation": relation,
            "anchor_utterance_index": anchor_index,
        }
        result.append(scheduled)
        previous_start = start
        speaker_ends[source["speaker_id"]] = end
    return result
