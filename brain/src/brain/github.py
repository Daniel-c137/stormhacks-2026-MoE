"""Read the team's GitHub repository's issues and pull requests through the GitHub MCP server.

Tool names and arguments follow GitHub's official MCP server; GITHUB_MCP_URL decides which server
answers. Every call goes through the read allowlist.
"""

from typing import Any, Literal
from urllib.parse import urlparse

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from brain.integrations import McpReader, McpToolError, ToolRefused, search_words
from brain.jira import root_cause

GitHubKind = Literal["issue", "pr"]
MAX_BODY = 400


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
    def __init__(self, repo: str, target: str | MCPServer):
        owner, _, name = repo.strip().partition("/")
        if not owner or not name or "/" in name:
            raise ValueError(f"the repository must be owner/name, not {repo!r}")
        self.owner, self.repo = owner, name
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
