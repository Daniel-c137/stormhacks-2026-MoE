"""Writes from the mock servers land in a journaled overlay. Reset runs outside MCP.

The journal is one JSON list per snapshot, world/.overlay/<snapshot>/journal.json (gitignored;
WORLD_OVERLAY_DIR moves it). The servers replay it over the mock data on every read, so a write
is seen by every later read and a reset takes effect at once, even while a server runs.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, TypeAdapter

from .config import Settings, SnapshotName

WriteOp = Literal["create_issue", "edit_issue", "transition_issue", "add_comment"]


class JournalEntry(BaseModel):
    at: datetime
    server: Literal["github", "jira"]
    tool: str
    op: WriteOp
    target: str
    arguments: dict[str, Any]

    @classmethod
    def now(
        cls,
        server: Literal["github", "jira"],
        tool: str,
        op: WriteOp,
        target: str,
        arguments: dict[str, Any],
    ) -> "JournalEntry":
        return cls(
            at=datetime.now(UTC),
            server=server,
            tool=tool,
            op=op,
            target=target,
            arguments=arguments,
        )


class WriteOverlay(Protocol):
    def record(self, entry: JournalEntry) -> None: ...

    def journal(self) -> list[JournalEntry]: ...

    def reset(self) -> None: ...


Journal = TypeAdapter(list[JournalEntry])
_lock = threading.Lock()  # tools run in worker threads; appends are read-modify-write


class JournalOverlay:
    """The write journal of one snapshot, as a JSON file under `root`."""

    def __init__(self, root: Path, snapshot: SnapshotName):
        self.path = root / snapshot / "journal.json"

    @classmethod
    def configured(cls) -> "JournalOverlay":
        settings = Settings()
        return cls(settings.world_overlay_dir, settings.world_snapshot)

    def record(self, entry: JournalEntry) -> None:
        with _lock:
            entries = [*self.journal(), entry]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_suffix(".tmp")
            partial.write_bytes(Journal.dump_json(entries, indent=2))
            partial.replace(self.path)

    def journal(self) -> list[JournalEntry]:
        try:
            return Journal.validate_json(self.path.read_bytes())
        except FileNotFoundError:
            return []

    def reset(self) -> None:
        with _lock:
            self.path.unlink(missing_ok=True)


def main(snapshot: SnapshotName | None = None) -> None:
    """CLI: clear the overlay between rehearsals: one snapshot's journal, or every snapshot's."""
    root = Settings().world_overlay_dir
    if snapshot is not None:
        overlay = JournalOverlay(root, snapshot)
        writes = len(overlay.journal())
        overlay.reset()
        print(f"Cleared the {snapshot} overlay ({writes} writes) in {overlay.path.parent}")
        return
    writes = 0
    for path in root.glob("*/journal.json"):
        writes += len(json.loads(path.read_text()))
        path.unlink()
    print(f"Cleared the world overlay ({writes} writes) in {root}")
