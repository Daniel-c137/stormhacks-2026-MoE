"""A repository's files at a snapshot: the main repository's in world/snapshots/<name>/repo/, at
the commit and branch in world/snapshots/<name>/repo.json; any other repository's in
world/snapshots/<name>/repos/<host>/<owner>/<repo>/, with <repo>.json next to it."""

import hashlib
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .config import SNAPSHOTS_DIR, SnapshotName


class RepoTree(BaseModel):
    branch: str  # the default branch, pointing at sha
    sha: str
    files: dict[str, str] = {}  # path from the repository root -> text


@cache
def load_repo(name: SnapshotName) -> RepoTree:
    """The main repository (world.json's github_repo)."""
    root = SNAPSHOTS_DIR / name
    return _tree(root / "repo", root / "repo.json")


@cache
def load_tree(name: SnapshotName, host: Literal["github", "gitlab"], full_name: str) -> RepoTree:
    """Another repository, or a GitLab project, by its owner/name or group/project path."""
    folder = SNAPSHOTS_DIR / name / "repos" / host / full_name
    return _tree(folder, folder.parent / f"{folder.name}.json")


def _tree(folder: Path, head: Path) -> RepoTree:
    tree = RepoTree.model_validate_json(head.read_text())
    files = {
        path.relative_to(folder).as_posix(): path.read_text()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }
    return tree.model_copy(update={"files": files})


def blob_sha(text: str) -> str:
    """The git blob SHA of a file's text, as GitHub reports it."""
    data = text.encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
