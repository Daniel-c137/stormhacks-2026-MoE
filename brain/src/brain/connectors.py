"""Whether GitHub and Jira can be read right now, checked against their MCP servers."""

import anyio
from mcp import Client

from contracts import ConnectorStatus, TeamSettings

from .config import Settings
from .integrations import McpServerName
from .jira import root_cause

# The read tools answers depend on. A server without them is reachable but not usable.
CORE_READ_TOOLS: dict[McpServerName, tuple[str, ...]] = {
    "github": ("issue_read", "search_issues", "list_pull_requests", "pull_request_read"),
    "jira": ("getJiraIssue", "searchJiraIssuesUsingJql"),
}


async def connector_statuses(config: Settings, team: TeamSettings) -> list[ConnectorStatus]:
    """GitHub then Jira, checked at the same time."""
    targets: dict[McpServerName, tuple[str | None, str, str | None, str]] = {
        "github": (config.github_mcp_url, "GITHUB_MCP_URL", team.github.repo, "repository"),
        "jira": (config.jira_mcp_url, "JIRA_MCP_URL", team.jira.project, "Jira project"),
    }
    found: dict[McpServerName, ConnectorStatus] = {}

    async def check(name: McpServerName) -> None:
        url, url_name, scope, scope_name = targets[name]
        if not url:
            found[name] = not_configured(name, f"Set {url_name} on the server")
        elif not scope:
            found[name] = not_configured(name, f"No {scope_name} is set in workspace settings")
        else:
            found[name] = await connector_status(name, url, timeout=config.connector_timeout)

    async with anyio.create_task_group() as group:
        for name in targets:
            group.start_soon(check, name)
    return [found[name] for name in targets]


async def connector_status(name: McpServerName, url: str, *, timeout: float) -> ConnectorStatus:
    """Connect, list the tools and check the core read tools are there."""
    try:
        with anyio.fail_after(timeout):
            async with Client(url) as client:
                tools = {tool.name for tool in (await client.list_tools()).tools}
    except TimeoutError:
        return failing(name, f"No answer from the MCP server: timed out after {timeout:g}s")
    except Exception as e:
        return failing(name, f"Could not reach the MCP server: {root_cause(e)}")
    if missing := [tool for tool in CORE_READ_TOOLS[name] if tool not in tools]:
        return failing(name, f"The MCP server is missing read tools: {', '.join(missing)}")
    return ConnectorStatus(name=name, state="connected")


def not_configured(name: McpServerName, detail: str) -> ConnectorStatus:
    return ConnectorStatus(name=name, state="not_configured", detail=detail)


def failing(name: McpServerName, detail: str) -> ConnectorStatus:
    return ConnectorStatus(name=name, state="failing", detail=detail[:300])
