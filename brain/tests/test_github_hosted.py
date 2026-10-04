"""Reading GitHub through an MCP server that wants a bearer token, as GitHub's hosted server
does, and what a pull request read carries: its checks and reviews when the server gives them."""

import logging
from typing import Any

import pytest
from github_cloud_support import OTHER_TOKEN, TOKEN, github_server, serve_with_token
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from brain.agent.team_tools import github_finding
from brain.connectors import connector_status
from brain.github import GitHubError, GitHubReader
from brain.integrations import McpEndpoint, McpReader, McpRejected

pytestmark = pytest.mark.anyio

REPO = "acme/checkout"


async def test_the_token_goes_as_a_bearer_header_on_every_request_with_read_only_asked():
    with serve_with_token(github_server("hosted"), TOKEN) as (url, seen):
        reader = GitHubReader(REPO, McpEndpoint(url, TOKEN))

        item = await reader.read(41, "issue")

    assert item.title == "Issue 41 from the hosted server"
    assert seen.tokens and set(seen.tokens) == {TOKEN}
    assert set(seen.readonly) == {"true"}


async def test_a_server_that_refuses_the_token_says_so_with_its_status():
    with serve_with_token(github_server("hosted"), TOKEN) as (url, _):
        reader = McpReader("github", McpEndpoint(url, OTHER_TOKEN))

        with pytest.raises(McpRejected) as refused:
            await reader.call("issue_read", {"method": "get", "owner": "acme", "repo": "x"})

    assert refused.value.status == 401
    assert OTHER_TOKEN not in str(refused.value)


async def test_a_reader_turns_the_refusal_into_a_github_error_without_the_token():
    with serve_with_token(github_server("hosted"), TOKEN) as (url, _):
        reader = GitHubReader(REPO, McpEndpoint(url, OTHER_TOKEN))

        with pytest.raises(GitHubError) as failed:
            await reader.read(41, "issue")

    assert "401" in str(failed.value)
    assert OTHER_TOKEN not in str(failed.value)


async def test_without_the_token_the_server_refuses(caplog):
    caplog.set_level(logging.DEBUG)
    with serve_with_token(github_server("hosted"), TOKEN) as (url, seen):
        status = await connector_status("github", url, timeout=5)
        with_token = await connector_status("github", McpEndpoint(url, TOKEN), timeout=5)

    assert status.state == "failing"
    assert with_token.state == "connected", with_token.detail
    assert None in seen.tokens and TOKEN in seen.tokens
    assert TOKEN not in caplog.text


def test_the_endpoint_never_shows_its_token():
    endpoint = McpEndpoint("https://api.githubcopilot.com/mcp/", TOKEN)

    assert TOKEN not in repr(endpoint) and TOKEN not in str(endpoint)
    assert "api.githubcopilot.com" in repr(endpoint)


# a pull request's checks and reviews


def pr_server(**answers: Any) -> MCPServer:
    """pull_request_read answering `get` with an open PR and each other method from `answers`
    (missing: an error, as the demo's mock answers reviews and check runs)."""
    server = MCPServer("github")

    @server.tool()
    def pull_request_read(method: str, owner: str, repo: str, pullNumber: int) -> Any:
        if method == "get":
            return {
                "number": pullNumber,
                "title": "Retry failed refunds",
                "state": "open",
                "html_url": f"https://github.com/{owner}/{repo}/pull/{pullNumber}",
                "body": "Retries a refund three times.",
            }
        if method not in answers:
            raise ToolError(f"{method} is not available")
        return answers[method]

    return server


async def test_a_pull_request_read_carries_its_checks_and_reviews():
    server = pr_server(
        get_status={
            "state": "failure",
            "total_count": 2,
            "statuses": [
                {"state": "success", "context": "ci/build"},
                {"state": "failure", "context": "ci/lint"},
            ],
        },
        get_check_runs={
            "total_count": 3,
            "check_runs": [
                {"name": "tests", "status": "completed", "conclusion": "success"},
                {"name": "e2e", "status": "in_progress", "conclusion": None},
                {"name": "docs", "status": "completed", "conclusion": "skipped"},
            ],
        },
        get_reviews=[
            {"user": {"login": "sam"}, "state": "CHANGES_REQUESTED"},
            {"user": {"login": "ada"}, "state": "COMMENTED"},
            {"user": {"login": "sam"}, "state": "APPROVED"},
            {"user": {"login": "lee"}, "state": "PENDING"},
        ],
    )
    reader = GitHubReader(REPO, server)

    item = await reader.read(77, "pr")
    finding = github_finding(REPO, item)

    assert item.checks == "2 passed, 1 failed (ci/lint), 1 pending"
    assert item.reviews == "sam approved, ada commented"
    assert finding.text == (
        f"Pull request {REPO}#77: Retry failed refunds (open); checks: 2 passed, 1 failed "
        "(ci/lint), 1 pending; reviews: sam approved, ada commented. Retries a refund three "
        "times."
    )
    assert finding.source.label == f"{REPO}#77"


async def test_a_pull_request_reads_without_checks_or_reviews_the_server_cannot_give():
    reader = GitHubReader(REPO, pr_server(get_status={"state": "pending", "total_count": 0}))

    item = await reader.read(77, "pr")

    assert (item.title, item.checks, item.reviews) == ("Retry failed refunds", None, None)
    assert github_finding(REPO, item).text == (
        f"Pull request {REPO}#77: Retry failed refunds (open). Retries a refund three times."
    )
