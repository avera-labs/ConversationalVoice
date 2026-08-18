from pathlib import Path

from voice_pipeline_quality_filter_audio_part.workspace import TaskWorkspace, cleanup_orphaned_workspaces


def test_workspace_cleanup_and_orphan_scope(tmp_path: Path) -> None:
    workspace = TaskWorkspace(prefix="owned-", parent=tmp_path)
    root = workspace.paths.root
    assert workspace.chunk_path(2).name == "2.wav"
    workspace.close()
    assert not root.exists()
    orphan = tmp_path / "owned-orphan"
    other = tmp_path / "other"
    orphan.mkdir()
    other.mkdir()
    assert cleanup_orphaned_workspaces(prefix="owned-", parent=tmp_path) == 1
    assert other.exists()
