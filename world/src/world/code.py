"""The repository's files at a snapshot: world/snapshots/<name>/repo/, at the commit and branch
in world/snapshots/<name>/repo.json."""

import hashlib
from functools import cache

from pydantic import BaseModel

from .config import SNAPSHOTS_DIR, SnapshotName


class RepoTree(BaseModel):
    branch: str  # the default branch, pointing at sha
    sha: str
    files: dict[str, str] = {}  # path from the repository root -> text


@cache
def load_repo(name: SnapshotName) -> RepoTree:
    root = SNAPSHOTS_DIR / name
    head = RepoTree.model_validate_json((root / "repo.json").read_text())
    files = {
        path.relative_to(root / "repo").as_posix(): path.read_text()
        for path in sorted((root / "repo").rglob("*"))
        if path.is_file()
    }
    return head.model_copy(update={"files": files})


def blob_sha(text: str) -> str:
    """The git blob SHA of a file's text, as GitHub reports it."""
    data = text.encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
