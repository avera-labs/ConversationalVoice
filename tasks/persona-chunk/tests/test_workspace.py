from voice_pipeline_persona_chunk.workspace import (
    WORKSPACE_MARKER,
    cleanup_orphaned_workspaces,
)


def test_cleanup_uses_literal_prefix_without_touching_other_directories(tmp_path):
    owned = tmp_path / "persona-worker-123"
    unrelated = tmp_path / "other-worker-123"
    wildcard_name = tmp_path / "persona[worker]-123"
    for path in (owned, unrelated, wildcard_name):
        path.mkdir()
        (path / "artifact").write_text("test")
    (owned / WORKSPACE_MARKER).touch()

    cleanup_orphaned_workspaces(prefix="persona-worker-", parent=tmp_path)

    assert not owned.exists()
    assert unrelated.exists()
    assert wildcard_name.exists()


def test_cleanup_requires_workspace_ownership_marker(tmp_path):
    unmarked = tmp_path / "persona-worker-unmarked"
    unmarked.mkdir()

    cleanup_orphaned_workspaces(prefix="persona-worker-", parent=tmp_path)

    assert unmarked.exists()
