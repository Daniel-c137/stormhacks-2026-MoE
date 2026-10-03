"""Mock Jira MCP server over the world snapshot.

Tool names follow Atlassian's MCP server. Names and parameters are not yet verified against it.
Done means merged to main; production deploys only from release tags.
"""

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .config import Settings

server = MCPServer("jira")
READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)


@server.tool(annotations=READ)
def searchJiraIssuesUsingJql(
    cloudId: str,
    jql: str,
    fields: list[str] | None = None,
    maxResults: int | None = None,
    nextPageToken: str | None = None,
) -> dict[str, Any]:
    """Supports the JQL subset the agent needs: project, status, assignee, sprint, key, text ~."""
    raise NotImplementedError


@server.tool(annotations=READ)
def getJiraIssue(
    cloudId: str, issueIdOrKey: str, fields: list[str] | None = None
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def getTransitionsForJiraIssue(cloudId: str, issueIdOrKey: str) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=READ)
def lookupJiraAccountId(cloudId: str, searchString: str) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=WRITE)
def createJiraIssue(
    cloudId: str,
    projectKey: str,
    issueTypeName: str,
    summary: str,
    description: str | None = None,
    assignee_account_id: str | None = None,
    additional_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=WRITE)
def editJiraIssue(cloudId: str, issueIdOrKey: str, fields: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=WRITE)
def transitionJiraIssue(
    cloudId: str, issueIdOrKey: str, transition: dict[str, str]
) -> dict[str, Any]:
    raise NotImplementedError


@server.tool(annotations=WRITE)
def addCommentToJiraIssue(cloudId: str, issueIdOrKey: str, commentBody: str) -> dict[str, Any]:
    raise NotImplementedError


def main() -> None:
    server.run("streamable-http", port=Settings().world_jira_mcp_port)
