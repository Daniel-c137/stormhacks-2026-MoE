"""Writes the dev and demo snapshots from one history, so dev never contradicts demo."""

from .config import SnapshotName
from .model import Snapshot


def generate(name: SnapshotName) -> Snapshot:
    raise NotImplementedError


def check(snapshot: Snapshot) -> list[str]:
    """Consistency checks; returns failures. Dev must hold nothing after its data_until date."""
    raise NotImplementedError


def main() -> None:
    """CLI: generate both snapshots, run checks, write JSON under world/snapshots/."""
    raise NotImplementedError
