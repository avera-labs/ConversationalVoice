import json
from pathlib import Path

import pytest

from voice_pipeline_split_raw_audio_into_parts.vad_artifact import (
    build_vad_artifact,
    serialize_vad_artifact,
    write_vad_artifact,
)
from voice_pipeline_split_raw_audio_into_parts.wav_io import SAMPLE_RATE
from voice_pipeline_split_raw_audio_into_parts.windowing import FrameSpan


def test_vad_artifact_is_stable_and_time_ordered() -> None:
    document = build_vad_artifact(
        model="pyannote/segmentation-3.0",
        audio_frame_count=SAMPLE_RATE * 10,
        segments=[
            FrameSpan(SAMPLE_RATE * 5, SAMPLE_RATE * 7),
            FrameSpan(SAMPLE_RATE, SAMPLE_RATE * 3),
        ],
    )

    assert document == {
        "schema_version": 1,
        "model": "pyannote/segmentation-3.0",
        "audio_duration_ms": 10_000,
        "segments": [
            {
                "index": 0,
                "start_ms": 1_000,
                "end_ms": 3_000,
                "duration_ms": 2_000,
            },
            {
                "index": 1,
                "start_ms": 5_000,
                "end_ms": 7_000,
                "duration_ms": 2_000,
            },
        ],
    }
    assert serialize_vad_artifact(document) == (
        b'{"schema_version":1,"model":"pyannote/segmentation-3.0",'
        b'"audio_duration_ms":10000,"segments":[{"index":0,"start_ms":1000,'
        b'"end_ms":3000,"duration_ms":2000},{"index":1,"start_ms":5000,'
        b'"end_ms":7000,"duration_ms":2000}]}\n'
    )


def test_empty_vad_result_still_writes_an_artifact(tmp_path: Path) -> None:
    document = build_vad_artifact(
        model="pyannote/segmentation-3.0",
        audio_frame_count=0,
        segments=[],
    )
    output = tmp_path / "vad_segments.json"

    write_vad_artifact(output, document)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "model": "pyannote/segmentation-3.0",
        "audio_duration_ms": 0,
        "segments": [],
    }


@pytest.mark.parametrize(
    "segments",
    [
        [FrameSpan(0, 20), FrameSpan(10, 30)],
        [FrameSpan(0, SAMPLE_RATE + 1)],
    ],
)
def test_invalid_segment_contract_is_rejected(segments: list[FrameSpan]) -> None:
    with pytest.raises(ValueError):
        build_vad_artifact(
            model="pyannote/segmentation-3.0",
            audio_frame_count=SAMPLE_RATE,
            segments=segments,
        )
