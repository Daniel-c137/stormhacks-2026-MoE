"""After the meeting: the report page, task review and approval, and the decision and task lists."""

import asyncio
from datetime import date

import pytest
from api_support import ALEX, OUTSIDER, SARAH, create
from fastapi.testclient import TestClient

from brain.api.deps import get_jira_pusher
from brain.jira import JiraConfig, JiraPusher
from contracts import Decision, Report, ReportProgress, TaskDraft

CONFIG = JiraConfig(
    mcp_url="http://unused.invalid/mcp",
    cloud_id="https://dropsubs.atlassian.net",
    project_key="DS",
    base_url="https://dropsubs.atlassian.net",
)


def task(meeting_id: str, n: int, **changes) -> TaskDraft:
    return TaskDraft(
        id=f"{meeting_id}-task-{n}",
        meeting_id=meeting_id,
        title=f"Task {n}",
        t=float(n * 10),
        quote=f"Quote {n}",
    ).model_copy(update=changes)


def decision(meeting_id: str, n: int, text: str | None = None) -> Decision:
    return Decision(
        id=f"{meeting_id}-decision-{n}",
        meeting_id=meeting_id,
        text=text or f"Decision {n}",
        made_by=ALEX.id,
        t=float(n * 10),
        quote=f"We decided {n}",
    )


def processed(store, meeting: dict, *tasks: TaskDraft, decisions=(), status="needs_review"):
    """Save a report for the meeting and move it to `status`, as the pipeline would."""
    report = Report(
        meeting_id=meeting["id"], summary="Summary", tasks=list(tasks), decisions=list(decisions)
    )

    async def save():
        await store.save_report(report)
        await store.set_status(meeting["id"], status)

    asyncio.run(save())
    return report


def meeting_in_store(store, meeting_id: str):
    return asyncio.run(store.meeting(meeting_id))


@pytest.fixture
def pushing(app, fake_jira):
    """Pushes go to the in-process FakeJira."""
    app.dependency_overrides[get_jira_pusher] = lambda: (
        lambda: JiraPusher(CONFIG, target=fake_jira.server)
    )
    return fake_jira


def push(client: TestClient, meeting_id: str, *task_ids: str, approved_by: str = "Someone"):
    return client.post(
        f"/meetings/{meeting_id}/tasks/push",
        json={"task_ids": list(task_ids), "destination": "jira", "approved_by": approved_by},
    )


def patch(client: TestClient, path_meeting_id: str, draft: TaskDraft, /, **changes):
    body = draft.model_copy(update=changes).model_dump(mode="json")
    return client.patch(f"/meetings/{path_meeting_id}/tasks/{draft.id}", json=body)


# the report page


def test_teammates_read_the_report(client_as, store):
    meeting = create(client_as(ALEX))
    report = processed(
        store, meeting, task(meeting["id"], 1), decisions=[decision(meeting["id"], 1)]
    )

    response = client_as(SARAH).get(f"/meetings/{meeting['id']}/report")

    assert response.status_code == 200
    assert Report.model_validate(response.json()) == report


def test_report_is_404_until_it_exists(client_as):
    meeting = create(client_as(ALEX))

    assert client_as(ALEX).get(f"/meetings/{meeting['id']}/report").status_code == 404


def test_another_teams_report_is_404(client_as, store):
    meeting = create(client_as(ALEX))
    processed(store, meeting, task(meeting["id"], 1))

    assert client_as(OUTSIDER).get(f"/meetings/{meeting['id']}/report").status_code == 404


def test_teammates_read_the_report_progress(client_as, store):
    meeting = create(client_as(ALEX))
    progress = ReportProgress(
        meeting_id=meeting["id"], steps=["Summary", "Tasks"], current=1, done=False
    )
    asyncio.run(store.save_report_progress(progress))

    response = client_as(SARAH).get(f"/meetings/{meeting['id']}/report/progress")

    assert response.status_code == 200
    assert ReportProgress.model_validate(response.json()) == progress
    assert client_as(OUTSIDER).get(f"/meetings/{meeting['id']}/report/progress").status_code == 404


def test_progress_is_404_when_the_pipeline_never_started(client_as):
    meeting = create(client_as(ALEX))

    assert client_as(ALEX).get(f"/meetings/{meeting['id']}/report/progress").status_code == 404


# editing a task draft


def test_reviewer_edits_the_editable_fields(client_as, store):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = patch(
        client_as(ALEX),
        meeting["id"],
        draft,
        title="Refund the 14 users",
        description="Via Stripe",
        owner_id=SARAH.id,
        due=date(2026, 10, 7),
        include=False,
    )

    assert response.status_code == 200, response.text
    expected = draft.model_copy(
        update={
            "title": "Refund the 14 users",
            "description": "Via Stripe",
            "owner_id": SARAH.id,
            "due": date(2026, 10, 7),
            "include": False,
        }
    )
    assert TaskDraft.model_validate(response.json()) == expected
    assert asyncio.run(store.report(meeting["id"])).tasks == [expected]


def test_owner_and_due_can_be_cleared(client_as, store):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1, owner_id=SARAH.id, due=date(2026, 10, 7))
    processed(store, meeting, draft)

    response = patch(client_as(ALEX), meeting["id"], draft, owner_id=None, due=None)

    assert response.status_code == 200
    assert response.json()["owner_id"] is None
    assert response.json()["due"] is None


def test_fields_the_reviewer_cannot_change_are_kept(client_as, store):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = patch(
        client_as(ALEX),
        meeting["id"],
        draft,
        title="New title",
        meeting_id="elsewhere",
        quote="Words nobody said",
        t=999.0,
        key="DS-1",
        jira_status="done",
    )

    assert response.status_code == 200
    assert TaskDraft.model_validate(response.json()) == draft.model_copy(
        update={"title": "New title"}
    )


def test_owner_must_be_on_the_team(client_as, store):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = patch(client_as(ALEX), meeting["id"], draft, owner_id=OUTSIDER.id)

    assert response.status_code == 422
    assert asyncio.run(store.report(meeting["id"])).tasks == [draft]


def test_title_cannot_be_blank(client_as, store):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    assert patch(client_as(ALEX), meeting["id"], draft, title="   ").status_code == 422


def test_a_pushed_task_cannot_be_edited(client_as, store):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1, key="DS-117", jira_status="todo")
    processed(store, meeting, draft, status="pushed")

    assert patch(client_as(ALEX), meeting["id"], draft, title="Changed").status_code == 409


def test_another_teams_task_is_404(client_as, store):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    assert patch(client_as(OUTSIDER), meeting["id"], draft, title="Mine").status_code == 404


def test_a_task_from_another_meeting_is_404(client_as, store):
    first = create(client_as(ALEX), "First")
    second = create(client_as(ALEX), "Second")
    draft = task(first["id"], 1)
    processed(store, first, draft)
    processed(store, second)

    assert patch(client_as(ALEX), second["id"], draft, title="Moved").status_code == 404


# approving the push


def test_approval_pushes_records_keys_and_marks_the_meeting_pushed(client_as, store, pushing):
    meeting = create(client_as(ALEX))
    one, two = task(meeting["id"], 1), task(meeting["id"], 2)
    processed(store, meeting, one, two)

    response = push(client_as(ALEX), meeting["id"], one.id, two.id)

    assert response.status_code == 200, response.text
    assert response.json() == [
        {
            "task_id": one.id,
            "key": "DS-117",
            "url": "https://dropsubs.atlassian.net/browse/DS-117",
            "error": None,
            "warning": None,
        },
        {
            "task_id": two.id,
            "key": "DS-118",
            "url": "https://dropsubs.atlassian.net/browse/DS-118",
            "error": None,
            "warning": None,
        },
    ]
    tasks = asyncio.run(store.report(meeting["id"])).tasks
    assert [(t.key, t.jira_status) for t in tasks] == [("DS-117", "todo"), ("DS-118", "todo")]
    saved = meeting_in_store(store, meeting["id"])
    assert saved.jira_keys == ["DS-117", "DS-118"]
    assert saved.status == "pushed"


def test_the_approver_is_the_caller_whatever_the_body_says(client_as, store, pushing):
    meeting = create(client_as(SARAH))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    push(client_as(SARAH), meeting["id"], draft.id, approved_by="The CEO")

    assert pushing.created[0]["description"].endswith(f"Approved for Jira by {SARAH.name}.")
    assert "The CEO" not in pushing.created[0]["description"]


def test_excluded_tasks_do_not_hold_the_meeting_in_review(client_as, store, pushing):
    meeting = create(client_as(ALEX))
    kept, dropped = task(meeting["id"], 1), task(meeting["id"], 2, include=False)
    processed(store, meeting, kept, dropped)

    push(client_as(ALEX), meeting["id"], kept.id)

    assert len(pushing.created) == 1
    assert meeting_in_store(store, meeting["id"]).status == "pushed"


def test_a_rejected_issue_leaves_the_meeting_in_review(client_as, store, pushing):
    meeting = create(client_as(ALEX))
    good, bad = task(meeting["id"], 1), task(meeting["id"], 2, title="FAIL this one")
    processed(store, meeting, good, bad)

    response = push(client_as(ALEX), meeting["id"], good.id, bad.id)

    assert response.status_code == 200
    results = response.json()
    assert results[0]["key"] == "DS-117"
    assert results[1]["key"] is None
    assert "summary" in results[1]["error"]
    tasks = {t.id: t for t in asyncio.run(store.report(meeting["id"])).tasks}
    assert tasks[good.id].key == "DS-117"
    assert tasks[bad.id].key is None
    saved = meeting_in_store(store, meeting["id"])
    assert saved.jira_keys == ["DS-117"]
    assert saved.status == "needs_review"


def test_retrying_after_a_fix_pushes_the_rest_and_finishes_review(client_as, store, pushing):
    meeting = create(client_as(ALEX))
    good, bad = task(meeting["id"], 1), task(meeting["id"], 2, title="FAIL this one")
    processed(store, meeting, good, bad)
    push(client_as(ALEX), meeting["id"], good.id, bad.id)

    assert patch(client_as(ALEX), meeting["id"], bad, title="Fixed").status_code == 200
    response = push(client_as(ALEX), meeting["id"], good.id, bad.id)

    assert [r["key"] for r in response.json()] == ["DS-117", "DS-118"]
    assert len(pushing.created) == 2
    saved = meeting_in_store(store, meeting["id"])
    assert saved.jira_keys == ["DS-117", "DS-118"]
    assert saved.status == "pushed"


@pytest.mark.parametrize("status", ["live", "processing"])
def test_push_is_refused_before_review(client_as, store, pushing, status):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft, status=status)

    assert push(client_as(ALEX), meeting["id"], draft.id).status_code == 409
    assert pushing.created == []


def test_push_to_another_teams_meeting_is_404(client_as, store, pushing):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    assert push(client_as(OUTSIDER), meeting["id"], draft.id).status_code == 404
    assert pushing.created == []


def test_push_without_jira_configured_is_503(client_as, store):
    meeting = create(client_as(ALEX))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = push(client_as(ALEX), meeting["id"], draft.id)

    assert response.status_code == 503
    assert "Jira is not configured" in response.json()["detail"]
    assert meeting_in_store(store, meeting["id"]).status == "needs_review"


# the decision and task lists


def test_decisions_are_the_teams_newest_first_and_searchable(client_as, store):
    older = create(client_as(ALEX), "Older")
    newer = create(client_as(ALEX), "Newer")
    processed(store, older, decisions=[decision(older["id"], 1, "Use Stripe for refunds")])
    processed(store, newer, decisions=[decision(newer["id"], 1, "Ship on Friday")])
    theirs = create(client_as(OUTSIDER), "Theirs")
    processed(store, theirs, decisions=[decision(theirs["id"], 1, "Use Stripe everywhere")])

    everything = client_as(SARAH).get("/decisions").json()
    stripe = client_as(SARAH).get("/decisions", params={"q": "stripe"}).json()

    assert [d["text"] for d in everything] == ["Ship on Friday", "Use Stripe for refunds"]
    assert [d["text"] for d in stripe] == ["Use Stripe for refunds"]


def test_tasks_are_the_teams_filtered_by_owner_and_open(client_as, store):
    meeting = create(client_as(ALEX))
    mine = task(meeting["id"], 1, owner_id=ALEX.id)
    done = task(meeting["id"], 2, owner_id=ALEX.id, key="DS-1", jira_status="done")
    hers = task(meeting["id"], 3, owner_id=SARAH.id)
    processed(store, meeting, mine, done, hers)
    theirs = create(client_as(OUTSIDER), "Theirs")
    processed(store, theirs, task(theirs["id"], 1))

    def ids(**params):
        return [t["id"] for t in client_as(ALEX).get("/tasks", params=params).json()]

    assert ids() == [mine.id, done.id, hers.id]
    assert ids(owner_id=ALEX.id) == [mine.id, done.id]
    assert ids(owner_id=ALEX.id, open="true") == [mine.id]
    assert ids(open="true") == [mine.id, hers.id]
    assert ids(open="false") == [mine.id, done.id, hers.id]
