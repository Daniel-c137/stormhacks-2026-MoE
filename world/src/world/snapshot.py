import copy
import json
from datetime import datetime
from functools import cache
from typing import Any, Literal

from .config import MOCK_DATA_DIR, SnapshotName
from .model import Snapshot


def mock_records(
    server: Literal["github", "gitlab", "jira"], kind: str, repo: str | None = None
) -> Any:
    """mock-data/<server>/<kind>.json as the real API returns it, e.g. ("jira", "issues"), or
    mock-data/<server>/<repo>/<kind>.json for one of several repositories. A fresh copy each
    call, so callers may apply overlay writes to it. The same records serve every snapshot."""
    return copy.deepcopy(_read(server, kind, repo))


@cache
def _read(server: str, kind: str, repo: str | None) -> Any:
    folder = MOCK_DATA_DIR / server / repo if repo else MOCK_DATA_DIR / server
    return json.loads((folder / f"{kind}.json").read_text())


def load_snapshot(name: SnapshotName) -> Snapshot:
    """Read world/snapshots/<name>/ with the write overlay applied on top."""
    raise NotImplementedError


def today(name: SnapshotName) -> datetime:
    """The agent's "today": WORLD_TODAY_OVERRIDE in rehearsals, else the snapshot's date."""
    raise NotImplementedError
