from voice_pipeline_reconstruct_chunk.timeline import schedule


def source(index, speaker, start, end):
    return {
        "utterance_index": index,
        "speaker_id": speaker,
        "source_start_ms": start,
        "source_end_ms": end,
    }


def test_schedule_preserves_gap_and_adapts_overlap_ratio():
    result = schedule(
        [
            source(0, 0, 100, 1100),
            source(1, 1, 600, 1300),
            source(2, 0, 1500, 1800),
        ],
        [2000, 400, 500],
    )
    assert (result[0]["start_ms"], result[0]["end_ms"]) == (100, 2100)
    assert result[1]["relation"] == "overlap"
    assert result[1]["start_ms"] == 1100
    assert result[2]["relation"] == "gap"
    assert result[2]["start_ms"] == 2100


def test_schedule_never_overlaps_same_speaker():
    result = schedule(
        [source(0, 0, 0, 1000), source(1, 0, 500, 1200)],
        [2000, 300],
    )
    assert result[1]["start_ms"] == result[0]["end_ms"]
