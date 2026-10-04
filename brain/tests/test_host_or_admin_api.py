"""What only a meeting's host could do, an admin may do too, so a meeting whose host left can still
be managed: ending it, its invitees and retrying its write-up. Approving its Jira push, the only
external write, is an admin's alone. Sarah hosts these meetings; Alex is the team's admin; Priya
is a member who is neither."""

import asyncio

import pytest
from api_support import ALEX, SARAH, TEAM, create
from test_report_review_api import CONFIG, processed, push, task

from brain.api.deps import get_jira_pusher
from brain.jira import JiraPusher
from contracts import Person

HOST_OR_ADMIN = "Only the host or an admin can do this"
ADMINS_ONLY = "Only an admin can do this"
PRIYA = Person(id="u-priya", name="Priya Natarajan", short="Priya", initials="PN")


@pytest.fixture(autouse=True)
def priya(store):
    asyncio.run(store.upsert_person(PRIYA, TEAM.id))


@pytest.fixture
def pushing(app, fake_jira):
    """Pushes go to the in-process FakeJira."""
    app.dependency_overrides[get_jira_pusher] = lambda: (
        lambda: JiraPusher(CONFIG, target=fake_jira.server)
    )
    return fake_jira


def status(client, meeting_id: str) -> str:
    return client.get(f"/meetings/{meeting_id}").json()["status"]


# ending a meeting


def test_an_admin_ends_a_meeting_its_host_left(client_as):
    meeting = create(client_as(SARAH))

    response = client_as(ALEX).post(f"/meetings/{meeting['id']}/end")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "processing"


def test_a_member_who_is_not_the_host_cannot_end_it(client_as):
    meeting = create(client_as(SARAH))

    response = client_as(PRIYA).post(f"/meetings/{meeting['id']}/end")

    assert response.status_code == 403
    assert response.json()["detail"] == HOST_OR_ADMIN
    assert status(client_as(SARAH), meeting["id"]) == "live"


# invitees


def test_an_admin_changes_the_invitees_of_someone_elses_meeting(client_as):
    meeting = create(client_as(SARAH))
    alex = client_as(ALEX)

    added = alex.post(f"/meetings/{meeting['id']}/invitees", json={"person_ids": [PRIYA.id]})
    assert added.status_code == 200, added.text
    assert added.json()["invitee_ids"] == [PRIYA.id]

    removed = alex.delete(f"/meetings/{meeting['id']}/invitees/{PRIYA.id}")
    assert removed.status_code == 200, removed.text
    assert removed.json()["invitee_ids"] == []


def test_a_member_who_is_not_the_host_cannot_change_invitees(client_as):
    meeting = create(client_as(SARAH))
    priya = client_as(PRIYA)

    added = priya.post(f"/meetings/{meeting['id']}/invitees", json={"person_ids": [ALEX.id]})
    removed = priya.delete(f"/meetings/{meeting['id']}/invitees/{ALEX.id}")

    assert (added.status_code, removed.status_code) == (403, 403)
    assert added.json()["detail"] == HOST_OR_ADMIN
    assert client_as(SARAH).get(f"/meetings/{meeting['id']}").json()["invitee_ids"] == []


# retrying the write-up


def test_an_admin_may_retry_the_write_up_and_a_member_may_not(client_as):
    meeting = create(client_as(SARAH))  # still live, so a permitted retry is refused as 409

    by_member = client_as(PRIYA).post(f"/meetings/{meeting['id']}/report/retry")
    by_admin = client_as(ALEX).post(f"/meetings/{meeting['id']}/report/retry")

    assert by_member.status_code == 403
    assert by_member.json()["detail"] == HOST_OR_ADMIN
    assert by_admin.status_code == 409
    assert "not being written up" in by_admin.json()["detail"]


# approving the Jira push, the only external write


@pytest.mark.parametrize("member", [SARAH, PRIYA], ids=["host", "member"])
def test_no_one_but_an_admin_approves_the_push(client_as, store, pushing, member):
    meeting = create(client_as(SARAH))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = push(client_as(member), meeting["id"], draft.id)

    assert response.status_code == 403
    assert response.json()["detail"] == ADMINS_ONLY
    assert pushing.created == []
    assert asyncio.run(store.meeting(meeting["id"])).status == "needs_review"


def test_an_admin_approves_and_is_recorded_as_the_approver(client_as, store, pushing):
    meeting = create(client_as(SARAH))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = push(client_as(ALEX), meeting["id"], draft.id, approved_by="Someone else")

    assert response.status_code == 200, response.text
    assert pushing.created[0]["description"].endswith(f"Approved for Jira by {ALEX.name}.")


def test_any_teammate_still_edits_task_drafts(client_as, store):
    meeting = create(client_as(SARAH))
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)
    body = draft.model_copy(update={"title": "Refund the users"}).model_dump(mode="json")

    response = client_as(PRIYA).patch(f"/meetings/{meeting['id']}/tasks/{draft.id}", json=body)

    assert response.status_code == 200, response.text
    assert response.json()["title"] == "Refund the users"
