"""Stand-ins for github.com, so no test reaches the network: GitHub's REST API (what the brain
asks when an admin connects a token) through an httpx transport, and GitHub's hosted MCP server
(what the team's reads go through) served on localhost, refusing any call without the token."""

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.responses import JSONResponse

API = "https://api.github.com"
LOGIN = "acme-reader"
# A made-up token in the fine-grained format; it opens nothing anywhere.
TOKEN = "github_pat_11TESTONLY0000000000_fakeTokenForTheBrainTestsOnly0000000000000000"
OTHER_TOKEN = "github_pat_11TESTONLY0000000000_anotherFakeTokenThatGitHubDoesNotKnow000000"

# What a fine-grained token's repository permissions let it read, by what the brain checks.
EVERYTHING = frozenset({"metadata", "contents", "issues", "pull_requests"})


class FakeGitHubApi:
    """Answers GET /user and, for each repository the token may read, the repository, one page
    of its issues, of its pull requests and its root folder; a part the token has no permission
    for answers 403 the way GitHub does, a repository it cannot see 404. Only
    https://api.github.com can be reached. A request without the token answers 401.

    To stage trouble: `down` (no connection), `failing` (every answer is a 502)."""

    def __init__(self, *, tokens: dict[str, str] | None = None):
        self.tokens = {TOKEN: LOGIN} if tokens is None else tokens
        self.repos: dict[str, frozenset[str]] = {}
        self.empty: set[str] = set()  # repositories with no commits: their root is a 404
        self.down = False
        self.failing = False
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self.handle)

    def allow(self, repo: str, permissions: frozenset[str] = EVERYTHING) -> None:
        self.repos[repo.casefold()] = permissions

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down or request.url.host != "api.github.com" or request.url.scheme != "https":
            raise httpx.ConnectError("no route to host", request=request)
        if self.failing:
            return httpx.Response(502, json={"message": "Server Error"})
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme != "Bearer" or token not in self.tokens:
            return httpx.Response(401, json={"message": "Bad credentials"})
        path = request.url.path
        if request.method != "GET":
            return httpx.Response(405, json={"message": "Not allowed"})
        if path == "/user":
            return httpx.Response(200, json={"login": self.tokens[token], "id": 1})
        parts = path.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "repos":
            return httpx.Response(404, json={"message": "Not Found"})
        repo = f"{parts[1]}/{parts[2]}".casefold()
        if repo not in self.repos:
            return httpx.Response(404, json={"message": "Not Found"})
        allowed = self.repos[repo]
        rest = parts[3:]
        needs = {
            (): "metadata",
            ("issues",): "issues",
            ("pulls",): "pull_requests",
            ("contents",): "contents",
        }.get(tuple(rest))
        if needs is None:
            return httpx.Response(404, json={"message": "Not Found"})
        if needs not in allowed:
            return httpx.Response(
                403, json={"message": "Resource not accessible by personal access token"}
            )
        if needs == "contents" and repo in self.empty:
            return httpx.Response(404, json={"message": "This repository is empty."})
        if needs == "metadata":
            return httpx.Response(200, json={"full_name": f"{parts[1]}/{parts[2]}"})
        return httpx.Response(200, json=[])

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


@dataclass
class Seen:
    """What reached a served MCP server: each request's bearer token (None without one) and its
    X-MCP-Readonly header."""

    tokens: list[str | None] = field(default_factory=list)
    readonly: list[str | None] = field(default_factory=list)


@contextmanager
def serve_with_token(server: MCPServer, token: str | None = TOKEN) -> Iterator[tuple[str, Seen]]:
    """`server` over streamable HTTP on localhost, like GitHub's hosted server: a request that
    does not carry `token` as a bearer header is refused with 401 before it reaches the server.
    With token=None nothing is required, like the demo's mock. Yields (url, what it saw)."""
    seen = Seen()
    inner = server.streamable_http_app()

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            scheme, _, given = headers.get("authorization", "").partition(" ")
            seen.tokens.append(given if scheme == "Bearer" else None)
            seen.readonly.append(headers.get("x-mcp-readonly"))
            if token is not None and (scheme != "Bearer" or given != token):
                response = JSONResponse({"message": "Bad credentials"}, status_code=401)
                await response(scope, receive, send)
                return
        await inner(scope, receive, send)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    http = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=http.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not http.started:
        if time.monotonic() > deadline:
            raise RuntimeError(f"fake MCP server {server.name} did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/mcp", seen
    finally:
        http.should_exit = True
        thread.join(timeout=10)


def github_server(label: str, repo: str = "acme/checkout") -> MCPServer:
    """A GitHub MCP server with the core read tools whose every answer names `label`, so a test
    can tell which server answered."""
    server = MCPServer(f"github-{label}")
    owner, name = repo.split("/")

    def issue(number: int) -> dict[str, Any]:
        return {
            "number": number,
            "title": f"Issue {number} from the {label} server",
            "state": "open",
            "html_url": f"https://github.com/{owner}/{name}/issues/{number}",
            "body": f"Read through the {label} server.",
        }

    @server.tool()
    def issue_read(method: str, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        return issue(issue_number)

    @server.tool()
    def search_issues(query: str, owner: str | None = None, repo: str | None = None) -> dict:
        return {"total_count": 1, "items": [issue(41)]}

    @server.tool()
    def list_pull_requests(owner: str, repo: str) -> list[dict[str, Any]]:
        return []

    @server.tool()
    def pull_request_read(method: str, owner: str, repo: str, pullNumber: int) -> dict:
        return {
            "number": pullNumber,
            "title": f"PR {pullNumber} from the {label} server",
            "state": "open",
            "html_url": f"https://github.com/{owner}/{name}/pull/{pullNumber}",
        }

    return server
