import socket
import threading
import time
from typing import Any

import pytest
import uvicorn
from api_support import app, client_as, settings, store, worker  # noqa: F401  (shared fixtures)
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeJira:
    """Stand-in for the Jira MCP server's createJiraIssue. A summary starting with FAIL is
    rejected the way Jira rejects an invalid field."""

    def __init__(self, first_number: int = 117):
        self.created: list[dict[str, Any]] = []
        self.server = MCPServer("jira")

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


@pytest.fixture
def jira_over_http():
    """FakeJira served over streamable HTTP on localhost: (fake, url)."""
    jira = FakeJira()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            jira.server.streamable_http_app(), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("fake Jira MCP server did not start")
        time.sleep(0.02)
    yield jira, f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def jira_env(jira_over_http, monkeypatch) -> FakeJira:
    """JIRA_* settings pointing at the FakeJira served on localhost."""
    jira, url = jira_over_http
    monkeypatch.setenv("JIRA_MCP_URL", url)
    monkeypatch.setenv("JIRA_PROJECT_KEY", "DS")
    monkeypatch.setenv("JIRA_BASE_URL", "https://dropsubs.atlassian.net")
    monkeypatch.delenv("JIRA_CLOUD_ID", raising=False)
    return jira
