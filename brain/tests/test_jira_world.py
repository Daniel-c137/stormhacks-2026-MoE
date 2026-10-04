"""The brain's Jira reads, ask tools, connector check and task push against the demo world's mock
Jira server over HTTP, the way JIRA_MCP_URL reaches it in the demo."""

from datetime import UTC, date, datetime

import pytest
from api_support import ALEX, TEAM
from conftest import serve_mcp

from brain.agent.team_tools import TeamToolbox
from brain.connectors import connector_status
from brain.jira import JiraConfig, JiraError, JiraPusher, JiraReader
from brain.report import ProcessedMeeting
from brain.store import InMemoryStore
from contracts import Person, Report, TaskDraft, TaskPushRequest
from world.jira_mcp import server as world_jira

pytestmark = pytest.mark.anyio

SITE = "https://dropsubs.atlassian.net"
REZA = Person(id="p-reza", name="Mohammad Reza", short="Reza", initials="MR")


@pytest.fixture
def world_url(tmp_path, monkeypatch):
    """The world's Jira server on localhost, writing to an overlay of its own."""
    monkeypatch.setenv("WORLD_OVERLAY_DIR", str(tmp_path / "overlay"))
    monkeypatch.setenv("WORLD_SNAPSHOT", "demo")
    with serve_mcp(world_jira) as url:
        yield url


@pytest.fixture
def config(world_url) -> JiraConfig:
    return JiraConfig(mcp_url=world_url, cloud_id=SITE, project_key="DS", base_url=SITE)


async def test_unfinished_work_comes_from_the_world_newest_first(config):
    issues = await JiraReader(config).unfinished()

    assert len(issues) == 10
    assert not any(issue.done for issue in issues)
    assert issues[0].key == "DS-116"  # updated last, on Oct 2
    assert "DS-61" not in {issue.key for issue in issues}  # done
    assert issues[0].url == f"{SITE}/browse/DS-116"


async def test_search_and_get_read_the_worlds_issues(config):
    reader = JiraReader(config)

    found = await reader.search("charged twice")
    issue = await reader.get("ds-104")

    assert [i.key for i in found] == ["DS-105", "DS-104"]
    assert issue.summary == "Subscriptions show up twice; fee charged twice"
    assert issue.status == "In Progress"
    assert issue.assignee == "Mohammad Reza"
    assert issue.priority
    assert issue.description and issue.description.startswith("Repeated last page")


async def test_a_missing_issue_is_a_jira_error(config):
    with pytest.raises(JiraError, match="DS-999"):
        await JiraReader(config).get("DS-999")


async def test_the_ask_tools_read_the_world(config):
    tools = TeamToolbox(
        TEAM.id,
        ALEX.id,
        InMemoryStore(),
        members=[],
        memory=None,
        jira=JiraReader(config),
        github="GitHub is not configured",
    )

    searched = await tools.call("jira_search", {"query": "double-charged users"})
    read = await tools.call("jira_issue", {"key": "DS-113"})

    assert searched.ok and [f.source.label for f in searched.content] == ["DS-105"]
    assert read.ok
    (finding,) = read.content
    assert finding.text.startswith("Jira DS-113: Charge 20% instead of 30%")
    assert finding.source.url == f"{SITE}/browse/DS-113"


async def test_the_connector_check_says_connected(world_url):
    status = await connector_status("jira", world_url, timeout=5)

    assert status.state == "connected", status.detail


async def test_an_approved_task_is_created_assigned_by_name_and_reads_back(config):
    task = TaskDraft(
        id="mtg-1-task-1",
        meeting_id="mtg-1",
        title="Email the 14 refunded users",
        owner_id=REZA.id,
        due=date(2026, 10, 7),
        quote="I'll email them once the refunds land.",
    )
    meeting = ProcessedMeeting(
        meeting_id="mtg-1",
        title="Friday standup",
        started_at=datetime(2026, 10, 2, 16, 30, tzinfo=UTC),
        members=[REZA],
        report=Report(meeting_id="mtg-1", summary="s", tasks=[task]),
    )
    request = TaskPushRequest(task_ids=[task.id], destination="jira", approved_by="Danial")

    (result,) = await JiraPusher(config).push(meeting, request)

    assert result.error is None and result.warning is None
    assert result.key == "DS-117"
    assert result.url == f"{SITE}/browse/DS-117"
    reader = JiraReader(config)
    issue = await reader.get("DS-117")
    assert issue.summary == "Email the 14 refunded users"
    assert issue.assignee == "Mohammad Reza"
    assert issue.status == "To Do" and not issue.done
    assert issue.description and "Approved for Jira by Danial." in issue.description
    assert "DS-117" in {i.key for i in await reader.unfinished(limit=50)}
