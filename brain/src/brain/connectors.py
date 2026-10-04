"""Whether GitHub, GitLab and Jira can be read right now, checked against their MCP servers, or
against a connected Jira account's own site."""

import anyio

from contracts import ConnectorStatus, TeamSettings

from .agent.team_tools import code_repos
from .config import Settings
from .github_account import Unusable, repo_target
from .integrations import McpEndpoint, McpRejected, McpServerName, McpTarget, mcp_client
from .jira import root_cause
from .jira_rest import JiraCloud, JiraRejected, JiraUnreachable

# The read tools answers depend on. A server without them is reachable but not usable.
CORE_READ_TOOLS: dict[McpServerName, tuple[str, ...]] = {
    "github": ("issue_read", "search_issues", "list_pull_requests", "pull_request_read"),
    "gitlab": ("search", "get_work_item", "get_merge_request"),
    "jira": ("getJiraIssue", "searchJiraIssuesUsingJql"),
}

Check = tuple[McpServerName, McpTarget]
NO_REPOSITORY = "No repository is set in workspace settings"
NOTHING_SET_UP = (
    "Connect a repository with its token in Settings, or set GITHUB_MCP_URL on the server"
)


async def connector_statuses(
    config: Settings,
    team: TeamSettings,
    github: dict[str, McpEndpoint | str] | None = None,
    jira: JiraCloud | str | None = None,
) -> list[ConnectorStatus]:
    """One per GitHub repository (`repo` names it), then GitLab, then Jira, checked at the same
    time. `github` are the repositories connected with a token (github_account.github_endpoints):
    each is checked on GitHub's hosted server with its own; the others share one check of
    GITHUB_MCP_URL, as GitLab's projects share one of its server. `jira` is the team's project
    on its own site when the agent reads it with the connected account, or why the account
    can't be used; None checks JIRA_MCP_URL."""
    # Each connector's status, or the key of the check that finds it: one check per server.
    found: dict[str, ConnectorStatus | object] = {}
    checks: dict[object, Check] = {}
    repos = code_repos(team, config)
    for repo in repos:
        try:
            target = repo_target(config, repo.path, github or {})
        except Unusable as e:
            detail = e.detail[:1].upper() + e.detail[1:]
            state = "failing" if e.failing else "not_configured"
            found[repo.path] = ConnectorStatus(name="github", state=state, detail=detail)
            continue
        key = ("github", target if isinstance(target, str) else id(target))
        checks[key] = ("github", target)
        found[repo.path] = key
    projects = ", ".join(p.path for p in team.gitlab.projects)
    if not config.gitlab_mcp_url:
        found["gitlab"] = not_configured("gitlab", "Set GITLAB_MCP_URL on the server")
    elif not projects:
        found["gitlab"] = not_configured("gitlab", "No GitLab project is set in workspace settings")
    else:
        found["gitlab"] = "gitlab"
        checks["gitlab"] = ("gitlab", config.gitlab_mcp_url)
    if isinstance(jira, str):
        found["jira"] = failing("jira", jira)
    elif jira is not None:
        found["jira"] = "jira-site"
    elif not config.jira_mcp_url:
        found["jira"] = not_configured("jira", "Set JIRA_MCP_URL on the server")
    elif not team.jira.project:
        found["jira"] = not_configured("jira", "No Jira project is set in workspace settings")
    else:
        found["jira"] = "jira"
        checks["jira"] = ("jira", config.jira_mcp_url)

    results: dict[object, ConnectorStatus] = {}

    async def check(key: object, name: McpServerName, target: McpTarget) -> None:
        results[key] = await connector_status(name, target, timeout=config.connector_timeout)

    async def check_site(cloud: JiraCloud) -> None:
        results["jira-site"] = await jira_site_status(cloud, timeout=config.connector_timeout)

    async with anyio.create_task_group() as group:
        for key, (name, target) in checks.items():
            group.start_soon(check, key, name, target)
        if isinstance(jira, JiraCloud):
            group.start_soon(check_site, jira)

    def status(entry: ConnectorStatus | object) -> ConnectorStatus:
        return entry if isinstance(entry, ConnectorStatus) else results[entry]

    github_statuses = [
        status(found[repo.path]).model_copy(update={"repo": repo.path}) for repo in repos
    ] or [not_configured("github", NO_REPOSITORY if config.github_mcp_url else NOTHING_SET_UP)]
    return [*github_statuses, status(found["gitlab"]), status(found["jira"])]


async def connector_status(
    name: McpServerName, target: McpTarget, *, timeout: float
) -> ConnectorStatus:
    """Connect, list the tools and check the core read tools are there."""
    try:
        with anyio.fail_after(timeout):
            async with mcp_client(target) as client:
                tools = {tool.name for tool in (await client.list_tools()).tools}
    except McpRejected as e:
        return failing(name, f"GitHub did not accept the token ({e.status})")
    except TimeoutError:
        return failing(name, f"No answer from the MCP server: timed out after {timeout:g}s")
    except Exception as e:
        return failing(name, f"Could not reach the MCP server: {root_cause(e)}")
    if missing := [tool for tool in CORE_READ_TOOLS[name] if tool not in tools]:
        return failing(name, f"The MCP server is missing read tools: {', '.join(missing)}")
    return ConnectorStatus(name=name, state="connected")


async def jira_site_status(cloud: JiraCloud, *, timeout: float) -> ConnectorStatus:
    """Whether the connected account still sees its project on its site."""
    site, project = cloud.access.site, cloud.access.project_key
    try:
        with anyio.fail_after(timeout):
            await cloud.project(project)
    except JiraRejected as e:
        if e.status in (401, 403):
            return failing("jira", f"{site} did not accept the account's email and API token")
        if e.status == 404:
            return failing("jira", f"The account can no longer see {project} on {site}")
        return failing("jira", f"{site} answered {e.status}: {e}")
    except TimeoutError:
        return failing("jira", f"No answer from {site}: timed out after {timeout:g}s")
    except JiraUnreachable as e:
        return failing("jira", str(e))
    return ConnectorStatus(name="jira", state="connected")


def not_configured(name: McpServerName, detail: str) -> ConnectorStatus:
    return ConnectorStatus(name=name, state="not_configured", detail=detail)


def failing(name: McpServerName, detail: str) -> ConnectorStatus:
    return ConnectorStatus(name=name, state="failing", detail=detail[:300])
