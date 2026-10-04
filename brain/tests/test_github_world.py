"""The brain's GitHub reads, ask tools, connector check and a fact-check against the demo world's
mock GitHub server over HTTP, the way GITHUB_MCP_URL reaches it in the demo."""

from datetime import UTC, date, datetime

import pytest
from api_support import ALEX, SARAH, TEAM
from conftest import serve_mcp
from fact_check_support import RELEASES_CALL, check, plan, say, scripted, verdict

from brain.agent.ask import PlannedCall
from brain.agent.factcheck import FactChecker, FactCheckVerdicts
from brain.agent.team_tools import TeamToolbox
from brain.config import Settings
from brain.connectors import connector_status
from brain.github import GitHubError, GitHubReader
from brain.store import InMemoryStore
from contracts import GitHubSettings, JiraSettings, Source, TeamSettings
from world.code import load_repo
from world.config import world_spec
from world.github_mcp import server as world_github

pytestmark = pytest.mark.anyio

REPO = world_spec().github_repo
WEB = f"https://github.com/{REPO}"
DEMO_HEAD = load_repo("demo").sha
RELEASED = "The Approve check fix from PR 50 is already released."


@pytest.fixture
def world_url(tmp_path, monkeypatch):
    """The world's GitHub server on localhost, writing to an overlay of its own."""
    monkeypatch.setenv("WORLD_OVERLAY_DIR", str(tmp_path / "overlay"))
    monkeypatch.setenv("WORLD_SNAPSHOT", "demo")
    with serve_mcp(world_github) as url:
        yield url


@pytest.fixture
def reader(world_url) -> GitHubReader:
    return GitHubReader(REPO, world_url)


async def test_the_connector_check_says_connected(world_url):
    status = await connector_status("github", world_url, timeout=5)

    assert status.state == "connected", status.detail


async def test_search_finds_the_worlds_issues_and_pull_requests(reader):
    issues = await reader.search("charged twice", "issue")
    prs = await reader.search("Approve check", "pr")

    assert [(i.kind, i.number) for i in issues] == [("issue", 43)]
    assert issues[0].title == "Subscriptions show up twice; fee charged twice"
    assert issues[0].state == "closed"
    assert issues[0].url == f"{WEB}/issues/43"
    assert [(p.kind, p.number) for p in prs] == [("pr", 50), ("pr", 27)]
    assert prs[0].url == f"{WEB}/pull/50"


async def test_a_search_no_item_has_every_word_of_finds_the_closest_first(reader):
    # PR 50 says "Approve check" but neither "approved" nor "fix"; the issue's test name is one
    # word (test_inbox_sync_pagination), so "inbox" and "sync" are not whole words in it
    prs = await reader.search("approved check fix", "pr")
    issues = await reader.search("flaky inbox sync test", "issue")

    assert prs[0].number == 50
    assert issues[0].number == 41


async def test_a_hyphenated_word_is_searched_as_its_words(reader):
    # speech-to-text writes "Approve-check"; PR 50 says "Approve check". DS-104 stays one word
    prs = await reader.search("Approve-check fix", "pr")

    assert prs[0].number == 50


async def test_a_search_that_matches_nothing_by_any_word_finds_nothing(reader):
    assert await reader.search("nonexistentword", "issue") == []
    assert await reader.search("nonexistentword anotherone", "pr") == []


async def test_the_latest_issues_and_pull_requests_come_most_recently_updated_first(reader):
    issues = await reader.latest("issue", 3)
    prs = await reader.latest("pr", 3)

    assert [(i.kind, i.number) for i in issues] == [("issue", 49), ("issue", 52), ("issue", 29)]
    assert issues[0].updated_at == datetime(2026, 10, 2, 23, 40, tzinfo=UTC)
    assert issues[0].url == f"{WEB}/issues/49"
    assert [(p.kind, p.number) for p in prs] == [("pr", 50), ("pr", 54), ("pr", 53)]


async def test_an_empty_search_lists_the_latest_with_when_each_was_updated(reader):
    tools = TeamToolbox(
        TEAM.id,
        ALEX.id,
        InMemoryStore(),
        members=[],
        memory=None,
        jira="Jira is not configured",
        github=reader,
    )

    latest = await tools.call("github_search", {"query": "", "kind": "issue"})
    latest_prs = await tools.call("github_search", {"query": "Latest pull requests", "kind": "pr"})
    topic = await tools.call("github_search", {"query": "latest renewal warning", "kind": "issue"})

    assert latest.ok
    assert [f.source.label for f in latest.content][:2] == [f"{REPO}#49", f"{REPO}#52"]
    assert "updated 2026-10-02" in latest.content[0].text
    assert [f.source.label for f in latest_prs.content][:2] == [f"{REPO}#50", f"{REPO}#54"]
    # a topic is searched for, not listed: no issue says "latest", so the closest comes first
    assert topic.content[0].source.label == f"{REPO}#36"


async def test_read_returns_merge_state_and_time(reader):
    merged = await reader.read(50, "pr")
    still_open = await reader.read(53, "pr")
    issue = await reader.read(36, "issue")

    assert merged.title == "Enforce the Approve check inside the cancel service"
    assert merged.merged is True
    assert merged.merged_at == datetime(2026, 10, 2, 23, 40, tzinfo=UTC)
    assert still_open.merged is False and still_open.merged_at is None
    assert still_open.state == "open"
    assert issue.kind == "issue" and issue.state == "open"
    assert issue.title == "Renewal warnings arrive after the charge"


async def test_a_missing_number_is_a_github_error(reader):
    with pytest.raises(GitHubError):
        await reader.read(999, "pr")


async def test_releases_come_newest_first_with_their_dates(reader):
    releases = await reader.releases()

    assert [r.tag for r in releases] == ["v0.9.3", "v0.9.1", "v0.8.0", "v0.7.0", "v0.6.0"]
    assert releases[0].published_at == datetime(2026, 9, 30, 22, tzinfo=UTC)
    assert releases[0].url == f"{WEB}/releases/tag/v0.9.3"


async def test_code_search_and_reads_still_work(reader):
    (hit,) = await reader.search_code("FEE_RATE")
    file = await reader.read_file(hit.path)

    assert hit.path == "api/billing/fees.py"
    assert file.pinned and file.ref == DEMO_HEAD
    assert "FEE_RATE" in file.text


async def test_the_ask_tools_read_the_world(reader):
    tools = TeamToolbox(
        TEAM.id,
        ALEX.id,
        InMemoryStore(),
        members=[],
        memory=None,
        jira="Jira is not configured",
        github=reader,
    )

    searched = await tools.call("github_search", {"query": "renewal warning", "kind": "issue"})
    read = await tools.call("github_read", {"number": 50, "kind": "pr"})
    releases = await tools.call("github_releases", {})

    assert searched.ok and [f.source.label for f in searched.content] == [f"{REPO}#36"]
    assert read.ok
    (finding,) = read.content
    assert "merged 2026-10-02" in finding.text
    assert finding.source == Source(kind="github_pr", label=f"{REPO}#50", url=f"{WEB}/pull/50")
    assert releases.ok
    latest = releases.content[0]
    assert latest.source.label == f"{REPO}@v0.9.3"
    assert latest.when == date(2026, 9, 30)


async def test_a_pr_merged_after_the_latest_release_is_flagged_as_not_released(store, world_url):
    await store.save_settings(
        TeamSettings(team_id=TEAM.id, github=GitHubSettings(repo=REPO), jira=JiraSettings())
    )
    meeting = await store.create_meeting(TEAM.id, "Launch sync", ALEX.id)
    await say(store, meeting, (SARAH, RELEASED, 10))
    llm = scripted(
        plan(check("c1", PlannedCall(tool="github_read", number=50, kind="pr"), RELEASES_CALL)),
        verdict(cite=(f"{REPO}#50", f"{REPO}@v0.9.3")),
    )
    checker = FactChecker(llm, store, settings=Settings(_env_file=None, github_mcp_url=world_url))

    response = await checker.tick(meeting, 60)

    [fact] = response.checks
    assert fact.claim == RELEASED
    assert (fact.verdict, fact.severity) == ("contradicted", "high")
    assert [s.label for s in fact.sources] == [f"{REPO}#50", f"{REPO}@v0.9.3"]
    [verdict_prompt] = [c.prompt for c in llm.calls if c.schema is FactCheckVerdicts]
    assert "merged 2026-10-02" in verdict_prompt
    assert "v0.9.3" in verdict_prompt and "2026-09-30" in verdict_prompt
