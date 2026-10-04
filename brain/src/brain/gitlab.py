"""Read a team's GitLab projects' issues, merge requests (with their diffs), releases and code
through GitLab's MCP server.

Tool names and arguments follow GitLab's official MCP server (GitLab 19.5, served by the GitLab
instance at /api/v4/mcp; docs.gitlab.com/user/model_context_protocol/mcp_server_tools): search,
get_work_item (issues are work items), get_merge_request (diffs with include ["diffs"]),
list_releases and get_repository_file. GITLAB_MCP_URL decides which server answers; every call
goes through the read allowlist, and nothing is written to GitLab.

The documentation names arguments but not every result's exact shape, so results are read
leniently: GitLab's REST field names (iid, web_url, merged_at) and GraphQL ones (webUrl,
mergedAt), at the top level or under one wrapping key. A result that is not one of this project's
is dropped.
"""

import base64
import binascii
import re
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlparse

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from brain.github import (
    COMMIT_SHA,
    MAX_BODY,
    CodeFile,
    CodeHit,
    CodeHost,
    CodeReadError,
    GitHubRelease,
    UnsafePath,
    branch_name,
    safe_path,
    timestamp,
)
from brain.integrations import McpReader, McpToolError, ToolRefused, search_words
from brain.jira import root_cause

GitLabKind = Literal["issue", "mr"]
GITLAB_WEB = "https://gitlab.com"
MCP_PATH = "/api/v4/mcp"
MAX_DIFF_FILES = 5
MAX_PATCH = 1200  # characters of one file's patch
PROJECT_PATH = re.compile(r"[\w.-]+(?:/[\w.-]+)+")


class GitLabError(CodeReadError):
    """A GitLab read failed."""


class GitLabItem(BaseModel):
    kind: GitLabKind
    number: int  # the iid, as the project numbers it
    title: str
    state: str | None = None  # opened, closed, merged, locked
    merged: bool | None = None
    merged_at: datetime | None = None
    body: str | None = None
    url: str | None = None


class GitLabDiff(BaseModel):
    """One changed file of a merge request: its path and its patch, clipped."""

    path: str
    status: str  # added, deleted, renamed or modified
    patch: str = ""


def web_address(mcp_url: str | None) -> str:
    """The instance's web address from its MCP endpoint (https://host/api/v4/mcp), else
    gitlab.com, e.g. for a mock served elsewhere."""
    if mcp_url and MCP_PATH in mcp_url:
        return mcp_url.split(MCP_PATH, 1)[0].rstrip("/")
    return GITLAB_WEB


def valid_project(path: str) -> str:
    """A project path, group/project with any subgroups; ValueError otherwise."""
    clean = path.strip().strip("/")
    if not PROJECT_PATH.fullmatch(clean) or any(p in (".", "..") for p in clean.split("/")):
        raise ValueError(f"the project must be group/project, not {path!r}")
    return clean


class GitLabReader:
    host: CodeHost = "gitlab"

    def __init__(
        self,
        project: str,
        target: str | MCPServer,
        ref: str | None = None,
        *,
        web: str = GITLAB_WEB,
    ):
        self.project = valid_project(project)
        self.ref = (ref or "").strip() or None  # None: the default branch
        self.web = web.rstrip("/")
        self.reader = McpReader("gitlab", target)

    @property
    def full_name(self) -> str:
        return self.project

    def label(self, kind: GitLabKind, number: int) -> str:
        """How GitLab writes a reference: group/project#7, or group/project!12 for an MR."""
        return f"{self.project}{'!' if kind == 'mr' else '#'}{number}"

    async def search(self, text: str, kind: GitLabKind, limit: int = 6) -> list[GitLabItem]:
        """Issues or merge requests of the project matching the text. Only plain words are
        sent, and anything from another project is dropped."""
        words = search_words(text)
        if not words:
            return []
        scope = "merge_requests" if kind == "mr" else "issues"
        data = await self._call(
            "search",
            {"scope": scope, "search": words, "project_id": self.project, "per_page": limit},
        )
        items = (self.item(raw, kind) for raw in records(data))
        return [item for item in items if item is not None and self.ours(item.url)][:limit]

    def ours(self, url: str | None) -> bool:
        """Whether a link is in this project: /group/project/-/... on the instance."""
        path = urlparse(url or "").path.casefold()
        return path.startswith(f"/{self.project}/-/".casefold())

    async def read(self, number: int, kind: GitLabKind) -> GitLabItem:
        """One issue (a work item) or merge request of the project, by iid."""
        if kind == "mr":
            arguments = {"project_id": self.project, "merge_request_iid": number}
            data = await self._call("get_merge_request", arguments)
        else:
            arguments = {"project_id": self.project, "work_item_iid": number}
            data = await self._call("get_work_item", arguments)
        raw = unwrap(data, ("merge_request", "work_item", "workItem", "result"))
        item = self.item(raw, kind) if raw is not None else None
        if item is None or (item.url and not self.ours(item.url)):
            raise GitLabError(f"GitLab returned nothing for {self.label(kind, number)}")
        return item

    async def diff(self, number: int, limit: int = MAX_DIFF_FILES) -> list[GitLabDiff]:
        """The files a merge request changes, with their patches clipped to MAX_PATCH."""
        data = await self._call(
            "get_merge_request",
            {
                "project_id": self.project,
                "merge_request_iid": number,
                "include": ["diffs"],
                "detail": "full_patch",
                "diffs_first": limit,
            },
        )
        found: list[GitLabDiff] = []
        for raw in diff_records(data)[:limit]:
            path = raw.get("new_path") or raw.get("newPath") or raw.get("old_path")
            if not isinstance(path, str) or not path:
                continue
            status = (
                "added"
                if raw.get("new_file") or raw.get("newFile")
                else "deleted"
                if raw.get("deleted_file") or raw.get("deletedFile")
                else "renamed"
                if raw.get("renamed_file") or raw.get("renamedFile")
                else "modified"
            )
            patch = raw.get("diff") if isinstance(raw.get("diff"), str) else ""
            if len(patch) > MAX_PATCH:
                patch = patch[: MAX_PATCH - 1] + "…"
            found.append(GitLabDiff(path=path, status=status, patch=patch))
        return found

    async def releases(self, limit: int = 5) -> list[GitHubRelease]:
        """The project's releases, most recently released first."""
        data = await self._call("list_releases", {"project_id": self.project, "per_page": limit})
        found = []
        for raw in records(data, ("releases",)):
            tag = raw.get("tag_name") or raw.get("tagName")
            if not isinstance(tag, str) or not tag.strip() or raw.get("upcoming") is True:
                continue
            body = raw.get("description")
            found.append(
                GitHubRelease(
                    tag=tag.strip(),
                    name=raw.get("name") if isinstance(raw.get("name"), str) else None,
                    published_at=timestamp(raw.get("released_at") or raw.get("releasedAt")),
                    body=body.strip()[:MAX_BODY]
                    if isinstance(body, str) and body.strip()
                    else None,
                    url=f"{self.web}/{self.project}/-/releases/{tag.strip()}",
                )
            )
        return found[:limit]

    async def search_code(self, text: str, limit: int = 5) -> list[CodeHit]:
        """Files of the project whose code matches the text (search's blobs scope)."""
        words = search_words(text)
        if not words:
            return []
        data = await self._call(
            "search",
            {"scope": "blobs", "search": words, "project_id": self.project, "per_page": limit},
        )
        hits: list[CodeHit] = []
        for raw in records(data):
            project = raw.get("project_id")
            if isinstance(project, str) and not project.isdigit() and not self.same(project):
                continue  # a numeric id is the project the search was scoped to
            try:
                hit = CodeHit(path=safe_path(str(raw.get("path") or raw.get("filename") or "")))
            except UnsafePath:
                continue
            if hit not in hits:
                hits.append(hit)
        return hits[:limit]

    def same(self, project: str) -> bool:
        return project.strip("/").casefold() == self.project.casefold()

    async def read_file(self, path: str, ref: str | None = None) -> CodeFile:
        """One text file of the project at `ref` (default: the team's ref, else HEAD), pinned to
        the commit the server names, if it names one."""
        path = safe_path(path)
        ref = (ref or "").strip() or self.ref
        data = await self._call(
            "get_repository_file",
            {"project_id": self.project, "file_path": path, "ref": ref or "HEAD"},
        )
        raw = unwrap(data, ("file", "result"))
        if isinstance(data, str):
            text, commit = data, None
        elif isinstance(raw, dict) and isinstance(raw.get("content"), str):
            text = raw["content"]
            if raw.get("encoding") == "base64":
                try:
                    text = base64.b64decode(text, validate=True).decode()
                except (binascii.Error, UnicodeDecodeError) as e:
                    raise GitLabError(f"{self.project}/{path} is not a text file") from e
            returned = raw.get("file_path") or raw.get("filePath")
            if isinstance(returned, str) and returned.strip("/") != path:
                raise GitLabError(f"GitLab returned another file for {self.project}/{path}")
            commit = next(
                (
                    raw[k]
                    for k in ("commit_id", "last_commit_id", "commitId")
                    if isinstance(raw.get(k), str) and COMMIT_SHA.fullmatch(raw[k])
                ),
                None,
            )
        else:
            raise GitLabError(f"GitLab returned no file for {self.project}/{path}")
        return CodeFile(
            repo=self.project,
            path=path,
            ref=commit or branch_name(ref),
            pinned=commit is not None,
            text=text,
            host="gitlab",
            web=self.web,
        )

    def item(self, raw: dict[str, Any], kind: GitLabKind) -> GitLabItem | None:
        number = raw.get("iid")
        if isinstance(number, str) and number.isdigit():
            number = int(number)
        if not isinstance(number, int) or isinstance(number, bool):
            return None
        state = raw.get("state")
        state = state.lower() if isinstance(state, str) else None
        if state == "open":
            state = "opened"  # GraphQL's OPEN
        merged_at = timestamp(raw.get("merged_at") or raw.get("mergedAt"))
        body = raw.get("description")
        url = raw.get("web_url") or raw.get("webUrl")
        return GitLabItem(
            kind=kind,
            number=number,
            title=str(raw.get("title") or "").strip(),
            state=state,
            merged=(state == "merged") if kind == "mr" and state else None,
            merged_at=merged_at,
            body=body.strip()[:MAX_BODY] if isinstance(body, str) and body.strip() else None,
            url=url if isinstance(url, str) else None,
        )

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        try:
            return await self.reader.call(tool, arguments)
        except ToolRefused:
            raise
        except McpToolError as e:
            raise GitLabError(str(e)) from e
        except Exception as e:
            raise GitLabError(f"GitLab MCP call failed: {root_cause(e)}") from e


def unwrap(data: Any, keys: tuple[str, ...]) -> Any:
    """The record itself, or the one under a wrapping key."""
    if isinstance(data, dict) and "iid" not in data and "content" not in data:
        for key in keys:
            if isinstance(data.get(key), dict):
                return data[key]
    return data


def records(data: Any, keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """A list result's records: the list itself, or under items, results, nodes or data."""
    for _ in range(2):
        if not isinstance(data, dict):
            break
        data = next(
            (
                data[k]
                for k in (*keys, "items", "results", "nodes", "data", "result")
                if isinstance(data.get(k), (list, dict))
            ),
            None,
        )
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def diff_records(data: Any) -> list[dict[str, Any]]:
    """A merge request's diffs: under diffs (a list, or a connection with nodes)."""
    if isinstance(data, dict):
        for holder in (data, data.get("merge_request"), data.get("result")):
            if isinstance(holder, dict) and "diffs" in holder:
                return records({"diffs": holder["diffs"]}, ("diffs",))
    return records(data)
