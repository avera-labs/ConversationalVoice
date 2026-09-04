from __future__ import annotations

import json

from voice_pipeline_score_completed_chunks.outputs import RunOutputs


def test_resume_canonicalizes_speaker_rows(tmp_path) -> None:
    path = tmp_path / "speaker-scores.jsonl"
    first = {
        "chunk_id": "one",
        "group": "reconstruction",
        "speaker_id": 0,
        "status": "failed",
    }
    second = {**first, "status": "success", "resume_key": "current"}
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    outputs = RunOutputs(tmp_path, resume=True)
    try:
        assert (
            outputs.existing_speakers[("one", "reconstruction", 0)]["status"]
            == "success"
        )
    finally:
        outputs.close()
    assert len(path.read_text().splitlines()) == 1


def test_resume_canonicalizes_audio_tag_rows(tmp_path) -> None:
    path = tmp_path / "audio-tag-scores.jsonl"
    first = {
        "chunk_id": "one",
        "group": "reconstruction",
        "transcript_index": 0,
        "status": "failed",
    }
    second = {**first, "status": "success", "resume_key": "current"}
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    outputs = RunOutputs(tmp_path, resume=True)
    try:
        assert (
            outputs.existing_audio_tags[("one", "reconstruction", 0)]["status"]
            == "success"
        )
    finally:
        outputs.close()
    assert len(path.read_text().splitlines()) == 1
