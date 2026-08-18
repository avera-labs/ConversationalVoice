from pathlib import Path

from voice_pipeline_diarize_audio_part.workspace import (
    TaskWorkspace,
    cleanup_orphaned_workspaces,
)


def test_workspace_context_removes_files(tmp_path: Path) -> None:
    workspace = TaskWorkspace(prefix="owned-", parent=tmp_path)
    root = workspace.paths.root
    assert workspace.paths.speaker_references.is_dir()
    assert workspace.speaker_reference_audio_path(3).name == "speaker-3.wav"
    assert workspace.paths.speaker_reference_manifest_path.name == "references.json"
    workspace.paths.audio_path.write_bytes(b"audio")
    workspace.close()
    assert not root.exists()


def test_startup_cleanup_is_namespaced_and_does_not_follow_symlinks(
    tmp_path: Path,
) -> None:
    orphan = tmp_path / "owned-one"
    orphan.mkdir()
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = tmp_path / "owned-link"
    symlink.symlink_to(outside, target_is_directory=True)

    assert cleanup_orphaned_workspaces(prefix="owned-", parent=tmp_path) == 1
    assert not orphan.exists()
    assert unrelated.exists()
    assert symlink.is_symlink()
    assert outside.exists()
