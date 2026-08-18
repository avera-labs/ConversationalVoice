from pathlib import Path

from voice_pipeline_separate_chunk.workspace import cleanup_orphaned_workspaces


def test_cleanup_is_prefix_scoped_and_ignores_symlinks(tmp_path: Path) -> None:
    owned = tmp_path / "owned-old"
    unrelated = tmp_path / "other-old"
    target = tmp_path / "target"
    symlink = tmp_path / "owned-link"
    owned.mkdir()
    unrelated.mkdir()
    target.mkdir()
    symlink.symlink_to(target, target_is_directory=True)

    assert cleanup_orphaned_workspaces(prefix="owned-", parent=tmp_path) == 1
    assert not owned.exists()
    assert unrelated.is_dir()
    assert symlink.is_symlink()
    assert target.is_dir()
