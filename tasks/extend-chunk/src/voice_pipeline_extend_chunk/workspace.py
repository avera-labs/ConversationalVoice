from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class Workspace:
    def __init__(self, prefix: str, parent: Path | None = None):
        self._temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=parent)
        self.root = Path(self._temporary.name)
        self.transcript = self.root / "input-transcript.json"
        self.reference_manifest = self.root / "references.json"
        self.script = self.root / "script.json"
        self.output_transcript = self.root / "transcript.json"
        self.reference_dir = self.root / "references"
        self.output_dir = self.root / "output"
        for directory in (self.reference_dir, self.output_dir):
            directory.mkdir()

    def reference(self, speaker_id: int) -> Path:
        return self.reference_dir / f"speaker-{speaker_id}.wav"

    def track(self, speaker_id: int) -> Path:
        return self.output_dir / f"speaker-{speaker_id}.wav"

    def close(self):
        self._temporary.cleanup()


def cleanup_orphaned_workspaces(*, prefix: str, parent: Path | None = None) -> None:
    base = parent or Path(tempfile.gettempdir())
    for candidate in base.glob(f"{prefix}*"):
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)
