from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class Workspace:
    def __init__(self, prefix: str, parent=None):
        self.root = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
        self.transcript = self.root / "source-transcript.json"
        self.reference_manifest = self.root / "references.json"
        self.manifest = self.root / "manifest.json"
        self.output_transcript = self.root / "transcript.json"

    def separated(self, slot: int) -> Path:
        return self.root / f"separated-{slot}.wav"

    def sample(self, slot: int) -> Path:
        return self.root / f"sample-{slot}.wav"

    def track(self, slot: int) -> Path:
        return self.root / f"speaker-{slot}.wav"

    def close(self):
        shutil.rmtree(self.root)


def cleanup_orphaned_workspaces(prefix: str, parent=None):
    base = Path(parent or tempfile.gettempdir())
    for path in base.glob(f"{prefix}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
