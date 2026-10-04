"""Mock GitHub MCP server over the world snapshot.

Tool names and parameters copy GitHub's official MCP server, so pointing GITHUB_MCP_URL at the
real server is a config change. Production is whatever the latest release contains.
"""

import json
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)

from .code import RepoTree, blob_sha, load_repo
from .config import Settings, world_spec

server = MCPServer("github")
READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)
CodeField = Literal["name", "path", "sha", "repository", "text_matches"]
ContentField = Literal[
    "type", "name", "path", "size", "sha", "url", "git_url", "html_url", "download_url"
]


def repo_tree() -> tuple[str, RepoTree]:
    """The world's repository (owner/name) and its files at the configured snapshot."""
    return world_spec().github_repo, load_repo(Settings().world_snapshot)


@server.tool(annotations=READ)
def search_code(
    query: str,
    sort: str | None = None,
    order: Literal["asc", "desc"] | None = None,
    page: int | None = None,
    perPage: int | None = None,
    fields: list[CodeField] | None = None,
) -> dict[str, Any]:
    """Code search over the default branch, in the official server's minimal result shape.
    Supports plain words (all must match the file's text or path, case-insensitively) and the
    repo:, org:, user:, path:, filename: and extension: qualifiers; other qualifiers and
    operators are ignored."""
    full_name, tree = repo_tree()
    owner = full_name.split("/")[0].casefold()
    words: list[str] = []
    paths: list[str] = []
    for token in query.split():
        key, colon, value = token.partition(":")
        key, value = key.casefold(), value.strip('"')
        if not colon:
            if token not in ("AND", "OR", "NOT"):
                words.append(token.strip('"').casefold())
        elif key == "repo" and value.casefold() != full_name.casefold():
            return {"total_count": 0, "incomplete_results": False, "items": []}
        elif key in ("org", "user") and value.casefold() != owner:
            return {"total_count": 0, "incomplete_results": False, "items": []}
        elif key == "path":
            paths.append(value.strip("/"))
        elif key == "filename":
            paths.append(f"*{value}")
        elif key == "extension":
            paths.append(f"*.{value.lstrip('.')}")

    def path_ok(path: str) -> bool:
        return all(path.endswith(p[1:]) if p.startswith("*") else path.startswith(p) for p in paths)

    items = [
        {
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "sha": blob_sha(text),
            "repository": full_name,
        }
        for path, text in tree.files.items()
        if path_ok(path) and all(w in path.casefold() or w in text.casefold() for w in words)
    ]
    size = perPage or 30
    start = ((page or 1) - 1) * size
    shown = items[start : start + size]
    if fields:
        shown = [{k: v for k, v in item.items() if k in fields} for item in shown]
    return {"total_count": len(items), "incomplete_results": False, "items": shown}


@server.tool(annotations=READ)
def get_file_contents(
    owner: str,
    repo: str,
    path: str = "/",
    ref: str | None = None,
    sha: str | None = None,
    fields: list[ContentField] | None = None,
) -> CallToolResult:
    """A file as a text message plus an embedded resource whose URI names the commit, or a
    directory as a JSON list of entries, like the official server. Only the default branch's
    head commit exists."""
    full_name, tree = repo_tree()
    if f"{owner}/{repo}".casefold() != full_name.casefold():
        raise ToolError(f"failed to get repository info: {owner}/{repo}: 404 Not Found")
    branch = (ref or tree.branch).removeprefix("refs/heads/")
    if (sha and sha != tree.sha) or branch != tree.branch:
        raise ToolError(f"failed to resolve git reference {sha or ref}: 404 Not Found")
    clean = path.strip().strip("/")
    if ".." in clean.split("/") or "\\" in clean:
        raise ToolError(f"invalid path {path!r}")

    if clean in tree.files:
        text = tree.files[clean]
        return CallToolResult(
            content=[
                TextContent(
                    type="text", text=f"successfully downloaded text file (SHA: {blob_sha(text)})"
                ),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"repo://{full_name}/sha/{tree.sha}/contents/{clean}",
                        mime_type="text/plain; charset=utf-8",
                        text=text,
                    ),
                ),
            ]
        )

    prefix = f"{clean}/" if clean else ""
    entries: dict[str, dict[str, Any]] = {}
    for file_path, text in tree.files.items():
        if not file_path.startswith(prefix):
            continue
        name, _, rest = file_path[len(prefix) :].partition("/")
        child = f"{prefix}{name}"
        entries.setdefault(
            child,
            {
                "type": "dir" if rest else "file",
                "name": name,
                "path": child,
                "size": 0 if rest else len(text.encode()),
                "sha": "" if rest else blob_sha(text),
                "html_url": f"https://github.com/{full_name}/{'tree' if rest else 'blob'}/"
                f"{tree.sha}/{child}",
            },
        )
    if not entries:
        raise ToolError(f"failed to get file contents {path}: 404 Not Found")
    listing = list(entries.values())
    if fields:
        listing = [{k: v for k, v in entry.items() if k in fields} for entry in listing]
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(listing))])


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
