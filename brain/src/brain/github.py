"""Read the team's GitHub repository's issues, pull requests and code through the GitHub MCP
server.

Tool names and arguments follow GitHub's official MCP server; GITHUB_MCP_URL decides which server
answers. Every call goes through the read allowlist.
"""

import re
from typing import Any, Literal
from urllib.parse import quote, unquote, urlparse

from mcp.server.mcpserver import MCPServer
from mcp.types import BlobResourceContents, EmbeddedResource, TextResourceContents
from pydantic import BaseModel

from brain.integrations import McpReader, McpToolError, ToolRefused, search_words
from brain.jira import root_cause

GitHubKind = Literal["issue", "pr"]
MAX_BODY = 400
MAX_CODE_QUERY = 200  # the server takes 256 characters, the repo: qualifier included
GITHUB_WEB = "https://github.com"

# Code search operators; even as plain words they could negate or widen the repo: scope.
OPERATORS = {"AND", "OR", "NOT"}
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
# The official server names a file it read repo://owner/repo/sha/<commit>/contents/<path>, or
# names the ref instead of a commit when it did not resolve one.
PINNED_FILE = re.compile(r"repo://([^/]+)/([^/]+)/sha/([0-9a-f]{40})/contents/(.+)")


class UnsafePath(ValueError):
    """A file path that could leave the repository: absolute, a URL, or with '..'."""


def safe_path(path: str) -> str:
    """A file path relative to the repository root, or UnsafePath."""
    clean = path.strip().removeprefix("./")
    parts = clean.split("/")
    if (
        not clean
        or clean.startswith("/")
        or "\\" in clean
        or "\0" in clean
        or ":" in parts[0]  # a URL scheme or a drive letter
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise UnsafePath(f"{path!r} is not a file path inside the repository")
    return clean


class CodeHit(BaseModel):
    """A file of the repository that code search matched."""

    path: str


class CodeFile(BaseModel):
    """A file's text as GitHub returned it, at `ref`: the commit SHA when the server resolved
    one (pinned), otherwise the branch or tag it was read at."""

    repo: str  # owner/name
    path: str
    ref: str
    pinned: bool
    text: str

    def lines(self) -> list[str]:
        return self.text.splitlines()

    def url(self, start_line: int, end_line: int) -> str:
        """The lines on GitHub; a permalink when the ref is a commit."""
        ref, path = quote(self.ref, safe="/"), quote(self.path, safe="/")
        return f"{GITHUB_WEB}/{self.repo}/blob/{ref}/{path}#L{start_line}-L{end_line}"


def branch_name(ref: str | None) -> str:
    """What a link names for a ref the server did not resolve: the branch or tag, or HEAD for
    the default branch."""
    if not ref:
        return "HEAD"
    return ref.removeprefix("refs/heads/").removeprefix("refs/tags/")


class GitHubError(RuntimeError):
    """A GitHub read failed."""


class GitHubItem(BaseModel):
    kind: GitHubKind
    number: int
    title: str
    state: str | None = None
    merged: bool | None = None
    body: str | None = None
    url: str | None = None


class GitHubReader:
    def __init__(self, repo: str, target: str | MCPServer, ref: str | None = None):
        owner, _, name = repo.strip().partition("/")
        if not owner or not name or "/" in name:
            raise ValueError(f"the repository must be owner/name, not {repo!r}")
        self.owner, self.repo = owner, name
        self.ref = (ref or "").strip() or None  # None: the default branch
        self.reader = McpReader("github", target)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    async def search(self, text: str, kind: GitHubKind, limit: int = 6) -> list[GitHubItem]:
        """Issues or pull requests of the repository matching the text.

        The server lets a repo:, org: or user: qualifier in the query replace the owner/repo
        scope, so only plain words are sent, and anything returned from elsewhere is dropped."""
        words = search_words(text)
        if not words:
            return []
        tool = "search_pull_requests" if kind == "pr" else "search_issues"
        data = await self._call(tool, {"query": words, "owner": self.owner, "repo": self.repo})
        if isinstance(data, dict):
            data = data.get("items", data.get("result"))
        if isinstance(data, dict):
            data = data.get("items")
        items = (self.item(raw, kind) for raw in data if isinstance(raw, dict)) if data else ()
        return [item for item in items if item is not None and self.ours(item.url)][:limit]

    def ours(self, url: str | None) -> bool:
        """Whether a result's link is in this repository: /owner/repo/... on any GitHub host."""
        path = urlparse(url or "").path.casefold()
        return path.startswith(f"/{self.owner}/{self.repo}/".casefold())

    async def read(self, number: int, kind: GitHubKind) -> GitHubItem:
        """One issue or pull request of the repository."""
        if kind == "pr":
            tool, arguments = "pull_request_read", {"pullNumber": number}
        else:
            tool, arguments = "issue_read", {"issue_number": number}
        data = await self._call(
            tool, {"method": "get", "owner": self.owner, "repo": self.repo, **arguments}
        )
        if isinstance(data, dict) and "number" not in data and isinstance(data.get("result"), dict):
            data = data["result"]
        item = self.item(data, kind) if isinstance(data, dict) else None
        if item is None:
            raise GitHubError(f"GitHub returned nothing for {self.full_name}#{number}")
        return item

    async def search_code(self, text: str, limit: int = 5) -> list[CodeHit]:
        """Files of the repository whose code matches the text.

        Code search takes no owner/repo arguments: the scope is the repo: qualifier added here
        to plain words, which cannot carry qualifiers or operators of their own. Results from
        any other repository are dropped."""
        words = " ".join(w for w in search_words(text).split() if w not in OPERATORS)
        if len(words) > MAX_CODE_QUERY:
            words = words[:MAX_CODE_QUERY].rsplit(" ", 1)[0]
        if not words:
            return []
        query = f"{words} repo:{self.full_name}"
        data = await self._call("search_code", {"query": query, "perPage": limit})
        hits: list[CodeHit] = []
        for raw in (data.get("items") if isinstance(data, dict) else None) or ():
            if not isinstance(raw, dict) or not self.same_repo(raw.get("repository")):
                continue
            try:
                hit = CodeHit(path=safe_path(str(raw.get("path") or "")))
            except UnsafePath:
                continue
            if hit not in hits:
                hits.append(hit)
        return hits[:limit]

    def same_repo(self, name: Any) -> bool:
        return isinstance(name, str) and name.casefold() == self.full_name.casefold()

    async def read_file(self, path: str, ref: str | None = None) -> CodeFile:
        """One text file of the repository at `ref` (default: the team's ref, else the default
        branch), pinned to the commit the server resolved the ref to when it names one."""
        path = safe_path(path)
        ref = (ref or "").strip() or self.ref
        arguments: dict[str, Any] = {"owner": self.owner, "repo": self.repo, "path": path}
        if ref:
            arguments["sha" if COMMIT_SHA.fullmatch(ref) else "ref"] = ref
        try:
            blocks = await self.reader.content("get_file_contents", arguments)
        except ToolRefused:
            raise
        except McpToolError as e:
            raise GitHubError(str(e)) from e
        except Exception as e:
            raise GitHubError(f"GitHub MCP call failed: {root_cause(e)}") from e

        resources = [b.resource for b in blocks if isinstance(b, EmbeddedResource)]
        if any(isinstance(r, BlobResourceContents) for r in resources):
            raise GitHubError(f"{self.full_name}/{path} is not a text file")
        texts = [r for r in resources if isinstance(r, TextResourceContents)]
        if not texts:
            raise GitHubError(f"GitHub returned no file for {self.full_name}/{path}")
        pinned = PINNED_FILE.fullmatch(str(texts[0].uri))
        if pinned and not (
            self.same_repo(f"{pinned.group(1)}/{pinned.group(2)}")
            and unquote(pinned.group(4)) == path
        ):
            raise GitHubError(f"GitHub returned another file for {self.full_name}/{path}")
        return CodeFile(
            repo=self.full_name,
            path=path,
            ref=pinned.group(3) if pinned else branch_name(ref),
            pinned=bool(pinned),
            text=texts[0].text,
        )

    def item(self, raw: dict[str, Any], kind: GitHubKind) -> GitHubItem | None:
        number = raw.get("number")
        if not isinstance(number, int):
            return None
        if kind == "issue" and raw.get("pull_request"):
            kind = "pr"  # issue search can return pull requests too
        body = raw.get("body")
        return GitHubItem(
            kind=kind,
            number=number,
            title=str(raw.get("title") or "").strip(),
            state=raw.get("state") if isinstance(raw.get("state"), str) else None,
            merged=raw.get("merged") if isinstance(raw.get("merged"), bool) else None,
            body=body.strip()[:MAX_BODY] if isinstance(body, str) and body.strip() else None,
            url=raw.get("html_url") if isinstance(raw.get("html_url"), str) else None,
        )

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        try:
            return await self.reader.call(tool, arguments)
        except ToolRefused:
            raise
        except McpToolError as e:
            raise GitHubError(str(e)) from e
        except Exception as e:
            raise GitHubError(f"GitHub MCP call failed: {root_cause(e)}") from e
