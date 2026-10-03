"""Writes from the mock servers land in a journaled overlay. Reset runs outside MCP."""

from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from .config import SnapshotName

WriteOp = Literal["create_issue", "edit_issue", "transition_issue", "add_comment"]


class JournalEntry(BaseModel):
    at: datetime
    server: Literal["github", "jira"]
    tool: str
    op: WriteOp
    target: str
    arguments: dict[str, Any]


class WriteOverlay(Protocol):
    def record(self, entry: JournalEntry) -> None: ...

    def journal(self) -> list[JournalEntry]: ...

    def reset(self) -> None: ...


def main(snapshot: SnapshotName | None = None) -> None:
    """CLI: clear the overlay between rehearsals."""
    raise NotImplementedError
