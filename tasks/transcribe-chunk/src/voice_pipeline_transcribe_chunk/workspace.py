from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class Workspace:
    def __init__(self, prefix: str, parent: Path | None = None):
        self.root = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
        self.speaker_paths = (self.root / "speaker-0.wav", self.root / "speaker-1.wav")
        self.transcript = self.root / "transcript.json"
        self.word_alignment = self.root / "word_alignment.json"

    def close(self):
        shutil.rmtree(self.root, ignore_errors=False)


def cleanup_orphaned_workspaces(*, prefix: str, parent: Path | None = None):
    base = parent or Path(tempfile.gettempdir())
    for path in base.glob(f"{prefix}*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
