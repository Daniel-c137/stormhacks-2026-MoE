from .config import SnapshotName


async def seed_meetings(name: SnapshotName) -> None:
    """Load the snapshot's past meetings into the brain's memory store."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError
