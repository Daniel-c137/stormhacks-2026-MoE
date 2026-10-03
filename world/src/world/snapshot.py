from datetime import datetime

from .config import SnapshotName
from .model import Snapshot


def load_snapshot(name: SnapshotName) -> Snapshot:
    """Read world/snapshots/<name>/ with the write overlay applied on top."""
    raise NotImplementedError


def today(name: SnapshotName) -> datetime:
    """The agent's "today": WORLD_TODAY_OVERRIDE in rehearsals, else the snapshot's date."""
    raise NotImplementedError
