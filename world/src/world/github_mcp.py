"""Mock GitHub MCP server over the world snapshot.

Tool names and parameters copy GitHub's official MCP server, so pointing GITHUB_MCP_URL at the
real server is a config change. Production is whatever the latest release contains.
"""

from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .config import Settings

server = MCPServer("github")
READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)


@server.tool(annotations=READ)
def list_issues(
    owner: str,
    repo: str,
    state: Literal["OPEN", "CLOSED"] | None = None,
    labels: list[str] | None = None,
    orderBy: Literal["CREATED_AT", "UPDATED_AT", "COMMENTS"] | None = None,
    direction: Literal["ASC", "DESC"] | None = None,
    since: str | None = None,
    perPage: int | None = None,
    after: str | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def search_issues(
    query: str,
    owner: str | None = None,
    repo: str | None = None,
    sort: str | None = None,
    order: Literal["asc", "desc"] | None = None,
    page: int | None = None,
    perPage: int | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def issue_read(
    method: Literal["get", "get_comments", "get_sub_issues", "get_parent", "get_labels"],
    owner: str,
    repo: str,
    issue_number: int,
    page: int | None = None,
    perPage: int | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def list_pull_requests(
    owner: str,
    repo: str,
    state: Literal["open", "closed", "all"] | None = None,
    head: str | None = None,
    base: str | None = None,
    sort: Literal["created", "updated", "popularity", "long-running"] | None = None,
    direction: Literal["asc", "desc"] | None = None,
    page: int | None = None,
    perPage: int | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def search_pull_requests(
    query: str,
    owner: str | None = None,
    repo: str | None = None,
    sort: str | None = None,
    order: Literal["asc", "desc"] | None = None,
    page: int | None = None,
    perPage: int | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def pull_request_read(
    method: Literal[
        "get",
        "get_diff",
        "get_status",
        "get_files",
        "get_commits",
        "get_review_comments",
        "get_reviews",
        "get_comments",
        "get_check_runs",
    ],
    owner: str,
    repo: str,
    pullNumber: int,
    page: int | None = None,
    perPage: int | None = None,
    after: str | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def list_commits(
    owner: str,
    repo: str,
    sha: str | None = None,
    author: str | None = None,
    path: str | None = None,
    since: str | None = None,
    until: str | None = None,
    page: int | None = None,
    perPage: int | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def list_releases(
    owner: str, repo: str, page: int | None = None, perPage: int | None = None
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def get_latest_release(owner: str, repo: str) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=WRITE)
def add_issue_comment(owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
    """Writes go to the overlay journal, never to the snapshot."""
    raise NotImplementedError


def main() -> None:
    server.run("streamable-http", port=Settings().world_github_mcp_port)
