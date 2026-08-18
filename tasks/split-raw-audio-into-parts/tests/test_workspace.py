from pathlib import Path

import pytest

from voice_pipeline_split_raw_audio_into_parts.workspace import task_workspace


def test_workspace_creates_paths_and_cleans_after_success(tmp_path: Path) -> None:
    with task_workspace(tmp_path) as workspace:
        root = workspace.root
        part_path = workspace.audio_part_path(3)
        workspace.raw_audio_path.write_bytes(b"raw")
        workspace.vad_segments_path.write_text("{}", encoding="utf-8")
        part_path.write_bytes(b"part")

        assert root.is_dir()
        assert workspace.audio_parts_directory.is_dir()
        assert part_path == root / "audio_parts" / "3" / "audio.wav"

    assert not root.exists()


def test_workspace_cleans_after_exception(tmp_path: Path) -> None:
    root: Path | None = None

    with pytest.raises(RuntimeError, match="test failure"):
        with task_workspace(tmp_path) as workspace:
            root = workspace.root
            workspace.raw_audio_path.write_bytes(b"partial")
            raise RuntimeError("test failure")

    assert root is not None
    assert not root.exists()


def test_workspace_rejects_negative_part_index(tmp_path: Path) -> None:
    with task_workspace(tmp_path) as workspace:
        with pytest.raises(ValueError, match="must not be negative"):
            workspace.audio_part_path(-1)
