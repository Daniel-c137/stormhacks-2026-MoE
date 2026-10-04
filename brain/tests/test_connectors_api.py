"""Connector status: GitHub and Jira reached through their MCP servers, never faked."""

import socket
from collections.abc import Iterator

import pytest
from api_support import ALEX, TEAM
from conftest import serve_mcp
from mcp.server.mcpserver import MCPServer


def read_server(name: str, *tools: str) -> MCPServer:
    server = MCPServer(name)
    for tool in tools:

        def read(query: str = "") -> dict:
            return {}

        server.add_tool(read, name=tool)
    return server


GITHUB_READ = ("issue_read", "search_issues", "list_pull_requests", "pull_request_read")
JIRA_READ = ("getJiraIssue", "searchJiraIssuesUsingJql")


@pytest.fixture
def github_url() -> Iterator[str]:
    with serve_mcp(read_server("github", *GITHUB_READ, "list_issues")) as url:
        yield url


@pytest.fixture
def jira_url() -> Iterator[str]:
    with serve_mcp(read_server("jira", *JIRA_READ)) as url:
        yield url


@pytest.fixture
def silent_url() -> Iterator[str]:
    """Accepts connections and never answers."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        yield f"http://127.0.0.1:{s.getsockname()[1]}/mcp"


@pytest.fixture
def closed_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/mcp"


@pytest.fixture
def linked(client_as):
    """The team has a repository and a Jira project."""
    body = {
        "team_id": TEAM.id,
        "github": {"repo": "acme/checkout", "ref": "main"},
        "jira": {"site": "acme.atlassian.net", "project": "DS"},
    }
    assert client_as(ALEX).put("/settings", json=body).status_code == 200


def statuses(client) -> dict[str, dict]:
    response = client.get("/settings/connectors")
    assert response.status_code == 200, response.text
    found = {s["name"]: s for s in response.json()}
    assert list(found) == ["github", "jira"]
    return found


def test_both_connect_when_their_servers_have_the_read_tools(
    client_as, settings, linked, github_url, jira_url
):
    settings.github_mcp_url, settings.jira_mcp_url = github_url, jira_url

    found = statuses(client_as(ALEX))

    assert found["github"]["state"] == "connected"
    assert found["jira"]["state"] == "connected"


def test_no_mcp_url_means_not_configured(client_as, settings):
    settings.github_mcp_url = settings.jira_mcp_url = None

    found = statuses(client_as(ALEX))

    assert found["github"]["state"] == "not_configured"
    assert "GITHUB_MCP_URL" in found["github"]["detail"]
    assert found["jira"]["state"] == "not_configured"
    assert "JIRA_MCP_URL" in found["jira"]["detail"]


def test_a_team_without_a_repo_or_project_is_not_configured(
    client_as, settings, github_url, jira_url
):
    settings.github_mcp_url, settings.jira_mcp_url = github_url, jira_url

    found = statuses(client_as(ALEX))

    assert found["github"]["state"] == "not_configured"
    assert "repository" in found["github"]["detail"]
    assert found["jira"]["state"] == "not_configured"
    assert "project" in found["jira"]["detail"]


def test_a_server_without_the_read_tools_is_failing(client_as, settings, linked, jira_over_http):
    _, url = jira_over_http  # FakeJira only creates issues
    settings.jira_mcp_url = url

    jira = statuses(client_as(ALEX))["jira"]

    assert jira["state"] == "failing"
    assert "getJiraIssue" in jira["detail"]


def test_an_unreachable_server_is_failing(client_as, settings, linked, closed_url):
    settings.github_mcp_url = closed_url

    github = statuses(client_as(ALEX))["github"]

    assert github["state"] == "failing"
    assert github["detail"]


def test_a_server_that_never_answers_times_out_as_failing(client_as, settings, linked, silent_url):
    settings.github_mcp_url = silent_url
    settings.connector_timeout = 0.5

    github = statuses(client_as(ALEX))["github"]

    assert github["state"] == "failing"
    assert "timed out" in github["detail"]
