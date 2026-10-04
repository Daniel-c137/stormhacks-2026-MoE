"""GitHub and Jira over MCP: the world mocks or the real servers, with the same tool names."""

import json
import re
from typing import Any, Literal

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent
from pydantic import BaseModel

McpServerName = Literal["github", "jira"]


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


class McpReader:
    """Calls one MCP server's read tools, and nothing else: anything off READ_TOOLS is refused
    before a connection is made."""

    def __init__(self, server: McpServerName, target: str | MCPServer):
        self.server = server
        self.target = target

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """The tool's structured result, or its text parsed as JSON, or the text itself."""
        if tool not in READ_TOOLS[self.server]:
            raise ToolRefused(f"{tool} is not a {self.server} read tool")
        async with Client(self.target) as client:
            result = await client.call_tool(tool, arguments)
        text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        if result.is_error:
            raise McpToolError(text or f"{tool} failed")
        if result.structured_content is not None:
            return result.structured_content
        try:
            return json.loads(text)
        except ValueError:
            return text
