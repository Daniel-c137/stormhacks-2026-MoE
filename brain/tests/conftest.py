import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import uvicorn
from api_support import app, client_as, settings, store, worker  # noqa: F401  (shared fixtures)
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from memory_support import memory_pool  # noqa: F401  (shared fixtures)
from pg_support import pg_dsn, pg_server  # noqa: F401  (shared fixtures)


@pytest.fixture
def anyio_backend():
    return "asyncio"


# The people a Jira site knows about. Names match the people in the test meetings.
JIRA_ACCOUNTS: list[dict[str, str]] = [
    {"accountId": "acc-alice", "displayName": "Alice Moreau", "emailAddress": "alice@dropsubs.dev"},
    {"accountId": "acc-bob", "displayName": "Bob Okafor", "emailAddress": "bob@dropsubs.dev"},
]


class FakeJira:
    """Stand-in for the Jira MCP server's createJiraIssue, lookupJiraAccountId and
    searchJiraIssuesUsingJql. A summary starting with FAIL is rejected the way Jira rejects an
    invalid field. Searches return `issues` (in Atlassian's shape) or fail with `search_error`.
    Lookups match any account whose name or email contains the search string; set
    `lookup_error` to make the lookup tool fail. Lookups answer {"users": [...]}, the shape the
    brain expects (unverified against Atlassian's server)."""

    def __init__(self, first_number: int = 117, accounts: list[dict[str, str]] | None = None):
        self.created: list[dict[str, Any]] = []
        self.accounts = list(JIRA_ACCOUNTS if accounts is None else accounts)
        self.lookups: list[str] = []
        self.lookup_error: str | None = None
        self.issues: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self.search_error: str | None = None
        self.server = MCPServer("jira")

        @self.server.tool()
        def lookupJiraAccountId(cloudId: str, searchString: str) -> dict[str, Any]:
            self.lookups.append(searchString)
            if self.lookup_error:
                raise ToolError(self.lookup_error)
            needle = searchString.strip().lower()
            users = [
                account
                for account in self.accounts
                if needle in account.get("displayName", "").lower()
                or needle in account.get("emailAddress", "").lower()
            ]
            return {"users": users}

        @self.server.tool()
        def searchJiraIssuesUsingJql(
            cloudId: str,
            jql: str,
            fields: list[str] | None = None,
            maxResults: int | None = None,
            nextPageToken: str | None = None,
        ) -> dict[str, Any]:
            self.searches.append(
                {"cloudId": cloudId, "jql": jql, "fields": fields, "maxResults": maxResults}
            )
            if self.search_error:
                raise ToolError(self.search_error)
            return {"issues": self.issues[: maxResults or None], "isLast": True}

        @self.server.tool()
        def createJiraIssue(
            cloudId: str,
            projectKey: str,
            issueTypeName: str,
            summary: str,
            description: str | None = None,
            assignee_account_id: str | None = None,
            additional_fields: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            if summary.startswith("FAIL"):
                raise ToolError("Field 'summary' is invalid")
            key = f"{projectKey}-{first_number + len(self.created)}"
            self.created.append(
                {
                    "cloudId": cloudId,
                    "projectKey": projectKey,
                    "issueTypeName": issueTypeName,
                    "summary": summary,
                    "description": description,
                    "assignee_account_id": assignee_account_id,
                    "additional_fields": additional_fields,
                    "key": key,
                }
            )
            return {"id": str(10000 + len(self.created)), "key": key}


@pytest.fixture
def fake_jira() -> FakeJira:
    return FakeJira()


@contextmanager
def serve_mcp(server: MCPServer) -> Iterator[str]:
    """Serve an in-process MCP server over streamable HTTP on localhost; yields its URL."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    http = uvicorn.Server(
        uvicorn.Config(
            server.streamable_http_app(), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=http.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not http.started:
        if time.monotonic() > deadline:
            raise RuntimeError(f"fake MCP server {server.name} did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        http.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def jira_over_http():
    """FakeJira served over streamable HTTP on localhost: (fake, url)."""
    jira = FakeJira()
    with serve_mcp(jira.server) as url:
        yield jira, url


@pytest.fixture
def jira_env(jira_over_http, monkeypatch) -> FakeJira:
    """JIRA_* settings pointing at the FakeJira served on localhost."""
    jira, url = jira_over_http
    monkeypatch.setenv("JIRA_MCP_URL", url)
    monkeypatch.setenv("JIRA_PROJECT_KEY", "DS")
    monkeypatch.setenv("JIRA_BASE_URL", "https://dropsubs.atlassian.net")
    monkeypatch.delenv("JIRA_CLOUD_ID", raising=False)
    return jira
