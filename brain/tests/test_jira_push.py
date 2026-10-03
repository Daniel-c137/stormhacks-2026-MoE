from datetime import date, datetime

import pytest

from brain.config import Settings
from brain.jira import (
    ApprovalRequired,
    JiraConfig,
    JiraPusher,
    JiraUnavailable,
    apply_results,
    jira_config,
)
from brain.report import ProcessedMeeting
from contracts import Person, Report, TaskDraft, TaskPushRequest, TaskPushResult

pytestmark = pytest.mark.anyio

CONFIG = JiraConfig(
    mcp_url="http://unused.invalid/mcp",
    cloud_id="https://dropsubs.atlassian.net",
    project_key="DS",
    base_url="https://dropsubs.atlassian.net",
)


def draft(n: int, **changes) -> TaskDraft:
    task = TaskDraft(
        id=f"mtg-standup-task-{n}",
        meeting_id="mtg-standup",
        title=f"Task {n}",
        t=6,
        quote="I'll refund the 14 affected users by Wednesday.",
    )
    return task.model_copy(update=changes)


def review(*tasks: TaskDraft) -> ProcessedMeeting:
    return ProcessedMeeting(
        meeting_id="mtg-standup",
        title="Friday standup",
        started_at=datetime(2026, 10, 2, 9, 30),
        members=[Person(id="p-bob", name="Bob Okafor", short="Bob", initials="BO")],
        report=Report(meeting_id="mtg-standup", summary="s", tasks=list(tasks)),
    )


def approve(*task_ids: str, approved_by: str = "Alice Moreau", destination: str = "jira"):
    return TaskPushRequest(
        task_ids=list(task_ids), destination=destination, approved_by=approved_by
    )


async def push(jira, meeting: ProcessedMeeting, request: TaskPushRequest):
    return await JiraPusher(CONFIG, target=jira.server).push(meeting, request)


async def test_approved_drafts_become_issues_with_keys_and_links(fake_jira):
    results = await push(fake_jira, review(draft(1), draft(2)), approve(draft(1).id, draft(2).id))

    assert results == [
        TaskPushResult(
            task_id=draft(1).id, key="DS-117", url="https://dropsubs.atlassian.net/browse/DS-117"
        ),
        TaskPushResult(
            task_id=draft(2).id, key="DS-118", url="https://dropsubs.atlassian.net/browse/DS-118"
        ),
    ]
    first = fake_jira.created[0]
    assert first["cloudId"] == "https://dropsubs.atlassian.net"
    assert first["projectKey"] == "DS"
    assert first["issueTypeName"] == "Task"
    assert first["summary"] == "Task 1"


async def test_the_issue_says_where_it_came_from_who_owns_it_and_when_it_is_due(fake_jira):
    task = draft(1, description="Refund via Stripe.", owner_id="p-bob", due=date(2026, 10, 7))

    await push(fake_jira, review(task, draft(2)), approve(task.id, draft(2).id))

    created, undated = fake_jira.created
    body = created["description"]
    assert "Refund via Stripe." in body
    assert '"Friday standup" (2026-10-02) at 00:06' in body
    assert "> I'll refund the 14 affected users by Wednesday." in body
    assert "Owner named in the meeting: Bob Okafor" in body
    assert "Approved for Jira by Alice Moreau" in body
    assert created["additional_fields"] == {"duedate": "2026-10-07"}
    assert undated["additional_fields"] is None
    assert created["assignee_account_id"] is None


async def test_nothing_is_pushed_without_an_approver(fake_jira):
    with pytest.raises(ApprovalRequired):
        await push(fake_jira, review(draft(1)), approve(draft(1).id, approved_by="  "))

    assert fake_jira.created == []


async def test_only_requested_drafts_that_are_still_included_are_pushed(fake_jira):
    meeting = review(draft(1), draft(2, include=False), draft(3))

    results = await push(fake_jira, meeting, approve(draft(1).id, draft(2).id, "task-404"))

    assert [r.key for r in results] == ["DS-117", None, None]
    assert "excluded" in results[1].error
    assert "no such task" in results[2].error.lower()
    assert [c["summary"] for c in fake_jira.created] == ["Task 1"]


async def test_a_draft_that_already_has_a_key_is_never_created_twice(fake_jira):
    results = await push(fake_jira, review(draft(1, key="DS-104")), approve(draft(1).id))

    assert results[0].key == "DS-104"
    assert results[0].url == "https://dropsubs.atlassian.net/browse/DS-104"
    assert fake_jira.created == []


async def test_one_rejected_issue_does_not_stop_the_others(fake_jira):
    meeting = review(draft(1, title="FAIL this one"), draft(2))

    results = await push(fake_jira, meeting, approve(draft(1).id, draft(2).id))

    assert results[0].key is None and "summary' is invalid" in results[0].error
    assert results[1].key == "DS-117"


async def test_other_destinations_are_refused_for_now(fake_jira):
    results = await push(fake_jira, review(draft(1)), approve(draft(1).id, destination="github"))

    assert results[0].key is None and "not supported" in results[0].error
    assert fake_jira.created == []


async def test_an_unreachable_server_is_a_result_not_a_crash():
    pusher = JiraPusher(CONFIG, target="http://127.0.0.1:9/mcp")

    results = await pusher.push(review(draft(1), draft(2)), approve(draft(1).id, draft(2).id))

    assert all(r.key is None for r in results)
    assert all("Jira MCP" in r.error and "connect" in r.error.lower() for r in results)


def test_apply_results_records_keys_on_the_pushed_drafts():
    tasks = [draft(1), draft(2), draft(3, key="DS-104", jira_status="in_progress")]
    results = [
        TaskPushResult(task_id=draft(1).id, key="DS-117"),
        TaskPushResult(task_id=draft(2).id, error="rejected"),
        TaskPushResult(task_id=draft(3).id, key="DS-104"),
    ]

    updated = apply_results(tasks, results)

    assert [(t.key, t.jira_status) for t in updated] == [
        ("DS-117", "todo"),
        (None, "draft"),
        ("DS-104", "in_progress"),
    ]


def test_jira_config_names_what_is_missing(monkeypatch):
    for name in ("JIRA_MCP_URL", "JIRA_PROJECT_KEY", "JIRA_CLOUD_ID", "JIRA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(JiraUnavailable) as error:
        jira_config(Settings(_env_file=None))

    for name in ("JIRA_MCP_URL", "JIRA_PROJECT_KEY", "JIRA_CLOUD_ID"):
        assert name in str(error.value)


def test_the_site_url_serves_as_cloud_id_when_none_is_given():
    config = jira_config(
        Settings(
            _env_file=None,
            jira_mcp_url="http://localhost:8102/mcp",
            jira_project_key="DS",
            jira_base_url="https://dropsubs.atlassian.net",
        )
    )

    assert config.cloud_id == "https://dropsubs.atlassian.net"
    assert config.base_url == "https://dropsubs.atlassian.net"
