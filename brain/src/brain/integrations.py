"""GitHub, GitLab and Jira over MCP: the world mocks or the real servers, with the same tool
names."""

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ContentBlock, TextContent
from pydantic import BaseModel

McpServerName = Literal["github", "gitlab", "jira"]


class McpServer(BaseModel):
    name: McpServerName
    url: str


# The orchestrator may call these on its own.
READ_TOOLS: dict[McpServerName, frozenset[str]] = {
    "github": frozenset(
        {
            "list_issues",
            "search_issues",
            "issue_read",
            "list_pull_requests",
            "search_pull_requests",
            "pull_request_read",
            "list_commits",
            "list_releases",
            "get_latest_release",
            "search_code",
            "get_file_contents",
        }
    ),
    # GitLab's official MCP server (GitLab 19.5): docs.gitlab.com/user/model_context_protocol/
    # mcp_server_tools. Issues are work items there.
    "gitlab": frozenset(
        {
            "search",
            "get_work_item",
            "list_work_items",
            "get_merge_request",
            "list_merge_requests",
            "get_repository_file",
            "list_releases",
        }
    ),
    "jira": frozenset(
        {
            "searchJiraIssuesUsingJql",
            "getJiraIssue",
            "getTransitionsForJiraIssue",
            "lookupJiraAccountId",
        }
    ),
}

# Only the approved task-push flow may call these.
WRITE_TOOLS: dict[McpServerName, frozenset[str]] = {
    "github": frozenset({"add_issue_comment"}),
    "gitlab": frozenset(),  # nothing is written to GitLab yet
    "jira": frozenset(
        {
            "createJiraIssue",
            "editJiraIssue",
            "transitionJiraIssue",
            "addCommentToJiraIssue",
        }
    ),
}


WORD = re.compile(r"[\w.]+(?:-[\w.]+)*")


def search_words(text: str) -> str:
    """Model-written search text as plain words: no quotes, operators or qualifiers such as
    repo: or project =, so it cannot widen a search beyond the scope the code sets. Hyphenated
    words like DS-104 stay whole."""
    return " ".join(WORD.findall(text))


class ToolRefused(PermissionError):
    """A tool that is not on the server's read allowlist."""


class McpToolError(RuntimeError):
    """The MCP server answered the call with an error."""


class McpRejected(RuntimeError):
    """The MCP server refused the credentials it was sent (HTTP 401 or 403)."""

    def __init__(self, status: int):
        super().__init__(f"the server did not accept the token ({status})")
        self.status = status


# The MCP SDK's own timeouts: a server may hold a response stream open.
MCP_TIMEOUT = httpx2.Timeout(30.0, read=300.0)


class McpEndpoint:
    """An MCP server over streamable HTTP that takes a bearer token, such as GitHub's hosted
    server with a team's personal access token. Only read-only tools are asked for
    (X-MCP-Readonly), on top of the read allowlist. The token is sent in the Authorization
    header and nowhere else; it is not part of what this object shows when printed."""

    def __init__(self, url: str, token: str):
        self.url = url
        self._token = token

    def __repr__(self) -> str:
        return f"McpEndpoint({self.url!r}, token=<hidden>)"

    __str__ = __repr__

    @asynccontextmanager
    async def client(self) -> AsyncIterator[Client]:
        """A connected client. McpRejected when the server refuses the token, which the MCP
        SDK alone would report as a bare error response."""
        refused: list[int] = []

        async def note(response: httpx2.Response) -> None:
            if response.status_code in (401, 403):
                refused.append(response.status_code)

        headers = {"Authorization": f"Bearer {self._token}", "X-MCP-Readonly": "true"}
        try:
            async with httpx2.AsyncClient(
                headers=headers, timeout=MCP_TIMEOUT, event_hooks={"response": [note]}
            ) as http:
                async with Client(streamable_http_client(self.url, http_client=http)) as client:
                    yield client
        except Exception as e:
            if refused:
                raise McpRejected(refused[0]) from e
            raise


McpTarget = str | MCPServer | McpEndpoint


@asynccontextmanager
async def mcp_client(target: McpTarget) -> AsyncIterator[Client]:
    """A connected client for a URL (no credentials), an in-process server or an endpoint that
    takes a token."""
    if isinstance(target, McpEndpoint):
        async with target.client() as client:
            yield client
    else:
        async with Client(target) as client:
            yield client


class McpReader:
    """Calls one MCP server's read tools, and nothing else: anything off READ_TOOLS is refused
    before a connection is made."""

    def __init__(self, server: McpServerName, target: McpTarget):
        self.server = server
        self.target = target

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """The tool's structured result, or its text parsed as JSON, or the text itself."""
        result = await self._call(tool, arguments)
        text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        if result.structured_content is not None:
            return result.structured_content
        try:
            return json.loads(text)
        except ValueError:
            return text

    async def content(self, tool: str, arguments: dict[str, Any]) -> list[ContentBlock]:
        """The tool's content blocks as they are, e.g. a file as an embedded resource."""
        return (await self._call(tool, arguments)).content

    async def _call(self, tool: str, arguments: dict[str, Any]) -> CallToolResult:
        if tool not in READ_TOOLS[self.server]:
            raise ToolRefused(f"{tool} is not a {self.server} read tool")
        async with mcp_client(self.target) as client:
            result = await client.call_tool(tool, arguments)
        if result.is_error:
            text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
            raise McpToolError(text or f"{tool} failed")
        return result
