from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

WORKSPACE_MARKER = ".voice-pipeline-persona-workspace"


class Workspace:
    def __init__(self, prefix: str, parent: Path | None = None):
        self.root = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
        try:
            (self.root / WORKSPACE_MARKER).touch(exist_ok=False)
        except Exception:
            shutil.rmtree(self.root, ignore_errors=True)
            raise
        self.audio = self.root / "chunk.wav"
        self.transcript = self.root / "transcript.json"
        self.mp3 = self.root / "chunk.mp3"
        self.persona = self.root / "persona.json"

    def close(self):
        shutil.rmtree(self.root, ignore_errors=False)


def cleanup_orphaned_workspaces(*, prefix: str, parent: Path | None = None):
    base = parent or Path(tempfile.gettempdir())
    for path in base.iterdir():
        marker = path / WORKSPACE_MARKER
        if (
            path.name.startswith(prefix)
            and path.is_dir()
            and not path.is_symlink()
            and marker.is_file()
            and not marker.is_symlink()
        ):
            shutil.rmtree(path)
