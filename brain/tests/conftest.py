import re
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
    searchJiraIssuesUsingJql, and with `issue_reads` getJiraIssue too. A summary starting with
    FAIL is rejected the way Jira rejects an invalid field. Searches return `issues` (in
    Atlassian's shape) or fail with `search_error`; reads find an issue in `issues` by key.
    Lookups match any account whose name or email contains the search string; set `lookup_error`
    to make the lookup tool fail. Lookups answer {"users": [...]}, the shape the brain expects
    (unverified against Atlassian's server)."""

    def __init__(
        self,
        first_number: int = 117,
        accounts: list[dict[str, str]] | None = None,
        *,
        issue_reads: bool = False,
    ):
        self.created: list[dict[str, Any]] = []
        self.accounts = list(JIRA_ACCOUNTS if accounts is None else accounts)
        self.lookups: list[str] = []
        self.lookup_error: str | None = None
        self.issues: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self.search_error: str | None = None
        self.reads: list[str] = []
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

        def getJiraIssue(
            cloudId: str, issueIdOrKey: str, fields: list[str] | None = None
        ) -> dict[str, Any]:
            self.reads.append(issueIdOrKey)
            for issue in self.issues:
                if issue["key"] == issueIdOrKey:
                    return issue
            raise ToolError(f"Issue {issueIdOrKey} does not exist")

        if issue_reads:
            self.server.tool()(getJiraIssue)

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


class FakeGitHub:
    """Stand-in for GitHub's MCP server: issue and pull request search and reads in GitHub's REST
    shapes, plus add_issue_comment, a write the agent must never call.

    Searches scope like the real server's prepareSearchArgs: a repo: qualifier in the query wins,
    otherwise owner/repo scope it; org: and user: narrow to an owner. Then every other word must
    be in the title. `elsewhere` holds items of other repositories the server's token can also
    read; `ignore_scope` makes searches return them regardless, like a misbehaving server.
    `releases` are the repository's releases, newest first, for list_releases and
    get_latest_release."""

    def __init__(self):
        self.ignore_scope = False
        self.elsewhere: list[dict[str, Any]] = [
            {
                "number": 7,
                "title": "Secret roadmap for the acquisition",
                "state": "open",
                "html_url": "https://github.com/otherorg/private-repo/issues/7",
                "body": "Confidential.",
            }
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.comments: list[dict[str, Any]] = []
        self.issues: dict[int, dict[str, Any]] = {
            41: {
                "number": 41,
                "title": "Waitlist email goes out before the exploit fix ships",
                "state": "open",
                "html_url": "https://github.com/dropsubs/app/issues/41",
                "body": "Hold the waitlist email until v0.9.4.",
            }
        }
        self.pulls: dict[int, dict[str, Any]] = {
            212: {
                "number": 212,
                "title": "Close the email exploit",
                "state": "closed",
                "merged": True,
                "html_url": "https://github.com/dropsubs/app/pull/212",
                "body": "Validates the signup address.",
            }
        }
        self.releases: list[dict[str, Any]] = []
        self.server = MCPServer("github")

        def search(
            items: dict[int, dict[str, Any]], query: str, owner: str | None, repo: str | None
        ) -> dict[str, Any]:
            scope = re.search(r"\brepo:(\S+)", query)
            prefix = scope.group(1) if scope else (f"{owner}/{repo}" if owner and repo else "")
            if not scope and (org := re.search(r"\b(?:org|user):(\S+)", query)):
                prefix = org.group(1)
            words = [w for w in query.lower().split() if ":" not in w]
            found = [
                i
                for i in [*items.values(), *self.elsewhere]
                if all(w in i["title"].lower() for w in words)
                and (self.ignore_scope or f"github.com/{prefix}/" in i["html_url"])
            ]
            return {"total_count": len(found), "items": found}

        @self.server.tool()
        def search_issues(
            query: str, owner: str | None = None, repo: str | None = None
        ) -> dict[str, Any]:
            self.calls.append(("search_issues", {"query": query, "owner": owner, "repo": repo}))
            return search(self.issues, query, owner, repo)

        @self.server.tool()
        def search_pull_requests(
            query: str, owner: str | None = None, repo: str | None = None
        ) -> dict[str, Any]:
            self.calls.append(
                ("search_pull_requests", {"query": query, "owner": owner, "repo": repo})
            )
            return search(self.pulls, query, owner, repo)

        @self.server.tool()
        def issue_read(method: str, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
            self.calls.append(
                ("issue_read", {"method": method, "owner": owner, "repo": repo, "n": issue_number})
            )
            if issue_number not in self.issues:
                raise ToolError("Not Found")
            return self.issues[issue_number]

        @self.server.tool()
        def pull_request_read(
            method: str, owner: str, repo: str, pullNumber: int
        ) -> dict[str, Any]:
            self.calls.append(
                (
                    "pull_request_read",
                    {"method": method, "owner": owner, "repo": repo, "n": pullNumber},
                )
            )
            if pullNumber not in self.pulls:
                raise ToolError("Not Found")
            return self.pulls[pullNumber]

        @self.server.tool()
        def list_releases(
            owner: str, repo: str, page: int | None = None, perPage: int | None = None
        ) -> list[dict[str, Any]]:
            self.calls.append(("list_releases", {"owner": owner, "repo": repo}))
            return self.releases[: perPage or None]

        @self.server.tool()
        def get_latest_release(owner: str, repo: str) -> dict[str, Any]:
            self.calls.append(("get_latest_release", {"owner": owner, "repo": repo}))
            if not self.releases:
                raise ToolError("Not Found")
            return self.releases[0]

        @self.server.tool()
        def add_issue_comment(owner: str, repo: str, issue_number: int, body: str) -> dict:
            self.comments.append({"issue_number": issue_number, "body": body})
            return {"id": 1}


@pytest.fixture
def fake_github() -> FakeGitHub:
    return FakeGitHub()


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
