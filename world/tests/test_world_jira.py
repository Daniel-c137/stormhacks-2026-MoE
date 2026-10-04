"""The mock Jira server answers from mock-data/jira/issues.json in Atlassian's shapes, journals its
writes to the overlay (never to mock-data) and forgets them on reset."""

import json

import pytest
from mcp import Client
from mcp.types import TextContent

from world import overlay
from world.config import MOCK_DATA_DIR
from world.jira_mcp import server
from world.overlay import JournalEntry, JournalOverlay
from world.snapshot import mock_records

pytestmark = pytest.mark.anyio

ISSUES_FILE = MOCK_DATA_DIR / "jira" / "issues.json"
SITE = "https://dropsubs.atlassian.net"
REZA = "712020:ds-mohammadreza"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def overlay_dir(tmp_path, monkeypatch):
    """Each test writes to its own overlay."""
    monkeypatch.setenv("WORLD_OVERLAY_DIR", str(tmp_path / "overlay"))
    monkeypatch.setenv("WORLD_SNAPSHOT", "demo")
    return tmp_path / "overlay"


def raw() -> list[dict]:
    return json.loads(ISSUES_FILE.read_text())


async def call(tool: str, **arguments):
    async with Client(server) as client:
        return await client.call_tool(tool, {"cloudId": SITE, **arguments})


def payload(result) -> dict:
    assert not result.is_error, text(result)
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(text(result))


def text(result) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


async def search(jql: str, **arguments) -> dict:
    return payload(await call("searchJiraIssuesUsingJql", jql=jql, **arguments))


async def keys(jql: str, **arguments) -> list[str]:
    return [issue["key"] for issue in (await search(jql, **arguments))["issues"]]


# searchJiraIssuesUsingJql


async def test_the_brains_unfinished_query_finds_every_issue_not_done_newest_first():
    jql = 'project = "DS" AND statusCategory != Done ORDER BY updated DESC'

    data = await search(jql, fields=["summary", "status", "assignee"], maxResults=50)

    expected = [i for i in raw() if i["fields"]["status"]["statusCategory"]["key"] != "done"]
    expected.sort(key=lambda i: i["fields"]["updated"], reverse=True)
    assert [i["key"] for i in data["issues"]] == [i["key"] for i in expected]
    assert len(expected) == 12
    assert data["isLast"] is True
    first = data["issues"][0]
    assert set(first["fields"]) == {"summary", "status", "assignee"}
    assert first["fields"]["status"]["statusCategory"]["key"] != "done"


async def test_the_brains_text_query_matches_every_word_in_summary_or_description():
    jql = 'project = "DS" AND text ~ "charged twice" ORDER BY updated DESC'

    # DS-105 says it in its description, DS-104 in its summary
    assert await keys(jql) == ["DS-105", "DS-104"]


async def test_text_search_ignores_case():
    assert await keys('project = DS AND text ~ "TESTFLIGHT"') == ["DS-64"]


async def test_max_results_pages_through_the_results():
    jql = "project = DS AND statusCategory != Done ORDER BY updated DESC"
    everything = await keys(jql)

    first = await search(jql, maxResults=5)
    second = await search(jql, maxResults=5, nextPageToken=first["nextPageToken"])

    assert [i["key"] for i in first["issues"]] == everything[:5]
    assert first["isLast"] is False
    assert [i["key"] for i in second["issues"]] == everything[5:10]


@pytest.mark.parametrize(
    ("jql", "expected"),
    [
        ("key = DS-104", ["DS-104"]),
        ('key = "ds-104"', ["DS-104"]),
        (
            'project = DS AND status = "In Progress" ORDER BY key',
            ["DS-101", "DS-102", "DS-104", "DS-106", "DS-109", "DS-111"],
        ),
        (
            'project = DS AND statusCategory = "To Do" ORDER BY key ASC',
            ["DS-105", "DS-107", "DS-113", "DS-114", "DS-115", "DS-116"],
        ),
        ("project = DS AND status = Backlog", ["DS-107"]),
        (
            'project = DS AND status != Done AND assignee = "Mohammad Reza" ORDER BY key',
            ["DS-104", "DS-105", "DS-106", "DS-113", "DS-114"],
        ),
        (
            f'assignee = "{REZA}" AND statusCategory = "To Do" ORDER BY key DESC',
            ["DS-114", "DS-113", "DS-105"],
        ),
        (
            "assignee = reyhaneh@dropsubs.example AND status = Done ORDER BY created",
            ["DS-61", "DS-62", "DS-69", "DS-64", "DS-63", "DS-103"],
        ),
        ("project = DS AND assignee IS EMPTY ORDER BY key", ["DS-107", "DS-115"]),
        (
            "project = DS AND assignee != Reyhaneh AND statusCategory = 'To Do' ORDER BY key",
            ["DS-105", "DS-113", "DS-114", "DS-116"],
        ),
        (
            "PROJECT = ds and STATUSCATEGORY != done order by KEY desc",
            [
                "DS-116",
                "DS-115",
                "DS-114",
                "DS-113",
                "DS-111",
                "DS-109",
                "DS-107",
                "DS-106",
                "DS-105",
                "DS-104",
                "DS-102",
                "DS-101",
            ],
        ),
    ],
)
async def test_the_jql_subset(jql, expected):
    assert await keys(jql, maxResults=50) == expected


@pytest.mark.parametrize(
    "jql",
    [
        "",
        "ORDER BY updated DESC",
        "project = DS OR project = OPS",
        "project = DS AND (status = Done)",
        "project = DS AND sprint in openSprints()",
        "project = DS AND priority = High",
        'project = DS AND summary ~ "fee"',
        "project = DS AND status IN (Done, Backlog)",
        "project = DS AND NOT status = Done",
        "project = DS AND created > -7d",
        'project = DS AND text ~ "unclosed',
        "project = DS AND status = Finished",
        "project = NOPE",
        "project = DS ORDER BY rank",
        "project = DS AND status = In Progress",
    ],
)
async def test_unsupported_or_invalid_jql_is_a_clear_error_not_everything(jql):
    result = await call("searchJiraIssuesUsingJql", jql=jql)

    assert result.is_error
    assert "JQL" in text(result) or "does not exist" in text(result)


async def test_a_bad_page_token_is_an_error():
    result = await call("searchJiraIssuesUsingJql", jql="project = DS", nextPageToken="nope")

    assert result.is_error


# getJiraIssue, getTransitionsForJiraIssue, lookupJiraAccountId


async def test_an_issue_reads_back_as_the_data_holds_it():
    issue = payload(await call("getJiraIssue", issueIdOrKey="DS-104"))

    (expected,) = [i for i in raw() if i["key"] == "DS-104"]
    assert issue == expected


async def test_an_issue_reads_by_id_or_lowercase_key_with_only_the_fields_asked_for():
    by_id = payload(await call("getJiraIssue", issueIdOrKey="10104", fields=["summary"]))
    by_key = payload(await call("getJiraIssue", issueIdOrKey="ds-104", fields=["summary"]))

    assert by_id == by_key
    assert by_id["key"] == "DS-104"
    assert by_id["fields"] == {"summary": "Subscriptions show up twice; fee charged twice"}


async def test_a_missing_issue_is_an_error():
    result = await call("getJiraIssue", issueIdOrKey="DS-999")

    assert result.is_error
    assert "DS-999" in text(result)


async def test_transitions_lead_to_every_other_status():
    data = payload(await call("getTransitionsForJiraIssue", issueIdOrKey="DS-105"))

    names = {t["name"] for t in data["transitions"]}
    assert names == {"Backlog", "In Progress", "Done"}
    done = next(t for t in data["transitions"] if t["name"] == "Done")
    assert done["to"]["statusCategory"]["key"] == "done"


async def test_lookup_finds_accounts_by_name_or_email():
    by_name = payload(await call("lookupJiraAccountId", searchString="Mohammad Reza"))
    by_email = payload(await call("lookupJiraAccountId", searchString="hossein@dropsubs.example"))
    nobody = payload(await call("lookupJiraAccountId", searchString="Alice"))

    assert [u["accountId"] for u in by_name["users"]] == [REZA]
    assert by_name["users"][0]["displayName"] == "Mohammad Reza"
    assert [u["displayName"] for u in by_email["users"]] == ["Hossein"]
    assert nobody == {"users": []}


async def test_lookup_knows_every_assignee_and_reporter():
    everyone = payload(await call("lookupJiraAccountId", searchString="dropsubs.example"))

    people = {
        person["accountId"]
        for issue in raw()
        for person in (issue["fields"]["assignee"], issue["fields"]["reporter"])
        if person
    }
    assert {u["accountId"] for u in everyone["users"]} == people


# Writes


async def test_a_created_issue_takes_the_next_key_and_reads_back(overlay_dir):
    before = ISSUES_FILE.read_bytes()

    created = payload(
        await call(
            "createJiraIssue",
            projectKey="DS",
            issueTypeName="Task",
            summary="Email the 14 refunded users",
            description="From the meeting.",
            assignee_account_id=REZA,
            additional_fields={"duedate": "2026-10-07"},
        )
    )

    assert created["key"] == "DS-117"
    assert created["id"] == "10117"
    issue = payload(await call("getJiraIssue", issueIdOrKey="DS-117"))
    fields = issue["fields"]
    assert fields["summary"] == "Email the 14 refunded users"
    assert fields["description"] == "From the meeting."
    assert fields["assignee"]["accountId"] == REZA
    assert fields["assignee"]["displayName"] == "Mohammad Reza"
    assert fields["issuetype"]["name"] == "Task"
    assert fields["project"]["key"] == "DS"
    assert fields["status"]["name"] == "To Do"
    assert fields["duedate"] == "2026-10-07"
    assert issue["url"] == f"{SITE}/browse/DS-117"
    assert "DS-117" in await keys("project = DS AND statusCategory != Done", maxResults=50)
    assert ISSUES_FILE.read_bytes() == before
    assert (overlay_dir / "demo" / "journal.json").exists()

    again = payload(
        await call("createJiraIssue", projectKey="DS", issueTypeName="Bug", summary="Another")
    )
    assert again["key"] == "DS-118"


@pytest.mark.parametrize(
    "arguments",
    [
        {"projectKey": "NOPE", "issueTypeName": "Task", "summary": "x"},
        {"projectKey": "DS", "issueTypeName": "Spaceship", "summary": "x"},
        {"projectKey": "DS", "issueTypeName": "Task", "summary": "  "},
        {"projectKey": "DS", "issueTypeName": "Task", "summary": "x", "assignee_account_id": "no"},
        {
            "projectKey": "DS",
            "issueTypeName": "Task",
            "summary": "x",
            "additional_fields": {"customfield_1": 3},
        },
        {
            "projectKey": "DS",
            "issueTypeName": "Task",
            "summary": "x",
            "additional_fields": {"duedate": "next week"},
        },
    ],
)
async def test_an_invalid_create_is_refused_and_nothing_is_written(arguments, overlay_dir):
    result = await call("createJiraIssue", **arguments)

    assert result.is_error
    assert JournalOverlay(overlay_dir, "demo").journal() == []


async def test_an_edit_reads_back():
    await call(
        "editJiraIssue",
        issueIdOrKey="DS-107",
        fields={"summary": "Connect a bank account", "assignee": {"accountId": REZA}},
    )

    fields = payload(await call("getJiraIssue", issueIdOrKey="DS-107"))["fields"]
    assert fields["summary"] == "Connect a bank account"
    assert fields["assignee"]["displayName"] == "Mohammad Reza"
    assert await keys('assignee = "Mohammad Reza" AND key = DS-107') == ["DS-107"]


@pytest.mark.parametrize(
    "fields",
    [{"status": {"name": "Done"}}, {"assignee": {"accountId": "nobody"}}, {"summary": ""}, {}],
)
async def test_an_invalid_edit_is_refused(fields):
    result = await call("editJiraIssue", issueIdOrKey="DS-107", fields=fields)

    assert result.is_error


async def test_a_transition_moves_the_issue_and_later_searches_see_it():
    transitions = payload(await call("getTransitionsForJiraIssue", issueIdOrKey="DS-105"))
    done = next(t for t in transitions["transitions"] if t["name"] == "Done")

    result = await call("transitionJiraIssue", issueIdOrKey="DS-105", transition={"id": done["id"]})

    assert not result.is_error, text(result)
    issue = payload(await call("getJiraIssue", issueIdOrKey="DS-105"))
    assert issue["fields"]["status"]["name"] == "Done"
    assert issue["changelog"]["histories"][-1]["items"] == [
        {"field": "status", "fromString": "To Do", "toString": "Done"}
    ]
    assert "DS-105" not in await keys("project = DS AND statusCategory != Done", maxResults=50)


async def test_an_unknown_transition_is_refused():
    result = await call("transitionJiraIssue", issueIdOrKey="DS-105", transition={"id": "999"})

    assert result.is_error


async def test_a_comment_reads_back():
    comment = payload(
        await call("addCommentToJiraIssue", issueIdOrKey="DS-104", commentBody="Fixed in #46.")
    )

    issue = payload(await call("getJiraIssue", issueIdOrKey="DS-104"))
    assert issue["fields"]["comment"]["total"] == 1
    (stored,) = issue["fields"]["comment"]["comments"]
    assert stored["body"] == "Fixed in #46."
    assert stored["id"] == comment["id"]


async def test_writes_to_a_missing_issue_are_refused():
    result = await call("addCommentToJiraIssue", issueIdOrKey="DS-999", commentBody="x")

    assert result.is_error


# Overlay and reset


async def test_reset_forgets_every_write(overlay_dir):
    await call("createJiraIssue", projectKey="DS", issueTypeName="Task", summary="Temporary")

    overlay.main()

    assert (await call("getJiraIssue", issueIdOrKey="DS-117")).is_error
    created = payload(
        await call("createJiraIssue", projectKey="DS", issueTypeName="Task", summary="Again")
    )
    assert created["key"] == "DS-117"


def test_the_journal_keeps_entries_in_order_per_snapshot(tmp_path):
    demo, dev = JournalOverlay(tmp_path, "demo"), JournalOverlay(tmp_path, "dev")
    first = JournalEntry.now("jira", "addCommentToJiraIssue", "add_comment", "DS-1", {"n": 1})
    second = JournalEntry.now("jira", "addCommentToJiraIssue", "add_comment", "DS-1", {"n": 2})

    demo.record(first)
    demo.record(second)

    assert JournalOverlay(tmp_path, "demo").journal() == [first, second]
    assert dev.journal() == []
    demo.reset()
    assert demo.journal() == []


def test_the_loader_returns_copies_of_the_mock_data():
    issues = mock_records("jira", "issues")
    issues[0]["fields"]["summary"] = "changed"

    assert mock_records("jira", "issues") == raw()
