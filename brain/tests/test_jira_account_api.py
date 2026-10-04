"""An admin connects the team's Jira account in Settings; approved task drafts then become real
issues on that site, pushed by an admin. The account's site and project are its own: the Jira
project the agent reads (Settings, Connectors) is left as it is. Alex is the team's admin; Sarah
is a member."""

import asyncio
from datetime import UTC, datetime

import anyio
import httpx
import pytest
from api_support import ALEX, AUTH_SECRET, OTHER_TEAM, OUTSIDER, SARAH, TEAM, create
from fastapi.testclient import TestClient
from jira_cloud_support import EMAIL, SITE, STORY, SUBTASK, TOKEN, FakeJiraCloud, text_of
from test_report_review_api import processed, push, task

from brain.api.deps import get_http_transport, get_settings
from brain.sealing import Unsealable, unseal
from brain.store import JiraAccount

ADMINS_ONLY = "Only an admin can do this"
CONNECT = {"site": SITE, "email": EMAIL, "api_token": TOKEN, "project": "DS"}
READ_FROM = {"site": "dropsubs.atlassian.net", "project": "DEMO"}  # the project the agent reads
NOT_CONNECTED = {
    "connected": False,
    "account_email": None,
    "account_site": None,
    "account_project": None,
}


@pytest.fixture
def jira(app) -> FakeJiraCloud:
    """The brain's outbound HTTP reaches only this Jira site."""
    fake = FakeJiraCloud()
    app.dependency_overrides[get_http_transport] = lambda: fake.transport
    return fake


def connect(client: TestClient, **changes):
    return client.put("/settings/jira/account", json=CONNECT | changes)


def account(store):
    return asyncio.run(store.jira_account(TEAM.id))


# connecting


def test_an_admin_connects_the_teams_jira_account(client_as, store, jira):
    response = connect(client_as(ALEX))

    assert response.status_code == 200, response.text
    assert response.json()["jira"] == {
        "site": None,
        "project": None,
        "connected": True,
        "account_email": EMAIL,
        "account_site": SITE,
        "account_project": "DS",
    }
    saved = account(store)
    assert (saved.site, saved.project, saved.email) == (SITE, "DS", EMAIL)
    assert saved.connected_by == ALEX.id
    assert [r.url.path for r in jira.requests] == [
        "/rest/api/3/myself",
        "/rest/api/3/project/DS",
        "/rest/api/3/mypermissions",
    ]


def test_the_token_is_kept_sealed_and_never_sent_back(client_as, store, jira):
    connected = connect(client_as(ALEX))
    seen_by_member = client_as(SARAH).get("/settings")

    saved = account(store)
    assert TOKEN not in saved.sealed_token
    assert unseal(saved.sealed_token, AUTH_SECRET, TEAM.id) == TOKEN
    with pytest.raises(Unsealable):  # copied into another team's row, it does not open
        unseal(saved.sealed_token, AUTH_SECRET, OTHER_TEAM.id)
    assert TOKEN not in connected.text and TOKEN not in seen_by_member.text
    assert seen_by_member.json()["jira"]["account_email"] == EMAIL


def test_connecting_leaves_the_project_the_agent_reads_as_it_is(client_as, jira):
    alex = client_as(ALEX)
    alex.put("/settings/connectors", json={"jira": READ_FROM})

    response = connect(alex)

    assert response.status_code == 200, response.text
    saved = response.json()["jira"]
    assert (saved["site"], saved["project"]) == (READ_FROM["site"], READ_FROM["project"])
    assert (saved["account_site"], saved["account_project"]) == (SITE, "DS")


def test_a_pasted_site_link_and_a_lowercase_key_are_tidied(client_as, jira):
    response = connect(client_as(ALEX), site=f"https://{SITE}/jira/software", project="ds")

    assert response.status_code == 200, response.text
    saved = response.json()["jira"]
    assert (saved["account_site"], saved["account_project"]) == (SITE, "DS")


def test_a_member_cannot_connect_or_disconnect(client_as, store, jira):
    sarah = client_as(SARAH)

    refused = connect(sarah)
    connect(client_as(ALEX))
    kept = sarah.delete("/settings/jira/account")

    assert (refused.status_code, kept.status_code) == (403, 403)
    assert refused.json()["detail"] == ADMINS_ONLY
    assert account(store) is not None


def test_a_wrong_email_or_token_is_refused_and_nothing_is_saved(client_as, store, jira):
    response = connect(client_as(ALEX), api_token="not-the-token")

    assert response.status_code == 422
    assert "email and API token" in response.json()["detail"]
    assert "not-the-token" not in response.text
    assert account(store) is None
    assert client_as(ALEX).get("/settings").json()["jira"]["connected"] is False


def test_a_project_the_account_cannot_see_is_refused(client_as, store, jira):
    response = connect(client_as(ALEX), project="NOPE")

    assert response.status_code == 422
    assert "NOPE" in response.json()["detail"]
    assert account(store) is None


def test_the_issue_type_tasks_are_created_as_is_chosen_when_connecting(client_as, store, jira):
    connect(client_as(ALEX))

    assert account(store).issue_type_id == "10001"  # the project's Task


def test_a_project_without_a_task_type_uses_its_first_ordinary_type(client_as, store, app):
    fake = FakeJiraCloud(issue_types=(SUBTASK, STORY))
    app.dependency_overrides[get_http_transport] = lambda: fake.transport

    response = connect(client_as(ALEX))

    assert response.status_code == 200, response.text
    assert account(store).issue_type_id == "10003"


def test_a_project_with_no_type_to_create_tasks_as_is_refused(client_as, store, app):
    fake = FakeJiraCloud(issue_types=(SUBTASK,))
    app.dependency_overrides[get_http_transport] = lambda: fake.transport

    response = connect(client_as(ALEX))

    assert response.status_code == 422
    assert "issue type" in response.json()["detail"]
    assert account(store) is None


def test_an_account_that_cannot_create_issues_in_the_project_is_refused(client_as, store, jira):
    jira.can_create = False

    response = connect(client_as(ALEX))

    assert response.status_code == 422
    assert "create issues" in response.json()["detail"]
    assert account(store) is None


def test_an_address_with_no_jira_site_says_so(client_as, store, jira):
    jira.missing = True

    response = connect(client_as(ALEX))

    assert response.status_code == 422
    assert response.json()["detail"] == f"There is no Jira site at {SITE}"
    assert account(store) is None


def test_a_request_the_brain_cannot_read_does_not_echo_the_token(client_as, jira):
    without_site = {k: v for k, v in CONNECT.items() if k != "site"}

    missing = client_as(ALEX).put("/settings/jira/account", json=without_site)
    mistyped = client_as(ALEX).put("/settings/jira/account", json=CONNECT | {"site": 7})

    assert (missing.status_code, mistyped.status_code) == (422, 422)
    assert TOKEN not in missing.text and TOKEN not in mistyped.text
    assert EMAIL in missing.text  # the rest of what was sent still shows what was wrong


@pytest.mark.parametrize("site", ["localhost:8080", "jira.internal.example", "169.254.169.254"])
def test_only_a_jira_cloud_site_is_ever_called(client_as, store, jira, site):
    response = connect(client_as(ALEX), site=site)

    assert response.status_code == 422
    assert "atlassian.net" in response.json()["detail"]
    assert jira.requests == [] and account(store) is None


@pytest.mark.parametrize("field", ["email", "api_token", "project"])
def test_every_field_is_needed(client_as, jira, field):
    response = connect(client_as(ALEX), **{field: "  "})

    assert response.status_code == 422
    assert jira.requests == []


def test_a_site_that_cannot_be_reached_is_a_502(client_as, store, jira):
    jira.down = True

    response = connect(client_as(ALEX))

    assert response.status_code == 502
    assert account(store) is None


def test_an_admin_disconnects_the_account(client_as, store, jira):
    alex = client_as(ALEX)
    connect(alex)

    response = alex.delete("/settings/jira/account")

    assert response.status_code == 200, response.text
    assert response.json()["jira"] == {"site": None, "project": None} | NOT_CONNECTED
    assert account(store) is None


def test_changing_or_removing_the_project_the_agent_reads_keeps_the_account(client_as, store, jira):
    alex = client_as(ALEX)
    connect(alex)

    changed = alex.put("/settings/connectors", json={"jira": READ_FROM})
    removed = alex.put("/settings/connectors", json={"jira": {"site": None, "project": None}})

    for response in (changed, removed):
        saved = response.json()["jira"]
        assert saved["connected"] is True
        assert (saved["account_site"], saved["account_project"]) == (SITE, "DS")
    assert changed.json()["jira"]["project"] == "DEMO"
    assert removed.json()["jira"]["project"] is None
    assert account(store) is not None


def test_what_settings_say_about_the_account_is_what_is_saved_for_it(client_as, store, jira):
    """The flags are read from the saved account, so they cannot disagree with what a push uses."""
    alex = client_as(ALEX)
    connect(alex)

    asyncio.run(store.delete_jira_account(TEAM.id))  # as if a disconnect stopped half-way
    assert alex.get("/settings").json()["jira"]["connected"] is False

    saved = JiraAccount(
        team_id=TEAM.id,
        site=SITE,
        project="OPS",
        issue_type_id=None,
        email="ops@acme.example",
        sealed_token="sealed",
        connected_by=ALEX.id,
        connected_at=datetime(2026, 10, 4, tzinfo=UTC),
    )
    asyncio.run(store.save_jira_account(saved))  # as if a connect stopped half-way
    shown = client_as(SARAH).get("/settings").json()["jira"]
    assert (shown["connected"], shown["account_project"], shown["account_email"]) == (
        True,
        "OPS",
        "ops@acme.example",
    )


def test_saving_other_settings_keeps_the_connection(client_as, store, jira):
    alex = client_as(ALEX)
    connected = connect(alex).json()

    saved = alex.put("/settings", json=connected | {"sensitivity": "quiet"})

    assert saved.status_code == 200, saved.text
    assert saved.json()["jira"]["connected"] is True
    assert account(store) is not None


def test_another_team_has_its_own_connection(client_as, jira):
    connect(client_as(ALEX))

    assert client_as(OUTSIDER).get("/settings").json()["jira"]["connected"] is False


# pushing


def test_an_admins_push_creates_real_issues_with_the_connected_account(client_as, store, jira):
    alex = client_as(ALEX)
    connect(alex)
    meeting = create(alex)
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = push(alex, meeting["id"], draft.id)

    assert response.status_code == 200, response.text
    assert response.json() == [
        {
            "task_id": draft.id,
            "key": "DS-1",
            "url": f"https://{SITE}/browse/DS-1",
            "error": None,
            "warning": None,
        }
    ]
    (created,) = jira.created
    assert created["summary"] == "Task 1"
    assert text_of(created["description"]).endswith(f"Approved for Jira by {ALEX.name}.")
    saved = alex.get(f"/meetings/{meeting['id']}/report").json()["tasks"][0]
    assert (saved["key"], saved["jira_status"]) == ("DS-1", "todo")
    assert saved["url"] == f"https://{SITE}/browse/DS-1"  # kept, so the link survives a reload
    assert alex.get(f"/meetings/{meeting['id']}").json()["status"] == "pushed"


def test_the_push_goes_to_the_accounts_project_not_the_one_the_agent_reads(client_as, store, jira):
    jira.projects.add("OPS")
    alex = client_as(ALEX)
    alex.put("/settings/connectors", json={"jira": READ_FROM})
    connect(alex, project="OPS")
    meeting = create(alex)
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    (result,) = push(alex, meeting["id"], draft.id).json()

    assert (result["key"], result["url"]) == ("OPS-1", f"https://{SITE}/browse/OPS-1")
    assert jira.created[0]["project"] == {"key": "OPS"}


def test_only_an_admin_pushes_even_for_their_own_meeting(client_as, store, jira):
    connect(client_as(ALEX))
    sarah = client_as(SARAH)
    meeting = create(sarah)
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = push(sarah, meeting["id"], draft.id)

    assert response.status_code == 403
    assert response.json()["detail"] == ADMINS_ONLY
    assert jira.created == []
    assert asyncio.run(store.meeting(meeting["id"])).status == "needs_review"


def test_without_a_connected_account_the_push_says_to_connect_one(client_as, store, jira):
    alex = client_as(ALEX)
    meeting = create(alex)
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = push(alex, meeting["id"], draft.id)

    assert response.status_code == 503
    assert "Settings" in response.json()["detail"]
    assert jira.requests == []


def test_a_token_sealed_with_an_old_server_secret_must_be_connected_again(
    app, client_as, store, settings, jira
):
    alex = client_as(ALEX)
    connect(alex)
    meeting = create(alex)
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)
    changed = settings.model_copy(update={"auth_secret": AUTH_SECRET + "-rotated"})
    app.dependency_overrides[get_settings] = lambda: changed

    response = push(alex, meeting["id"], draft.id)

    assert response.status_code == 409
    assert "Connect Jira again" in response.json()["detail"]
    assert jira.created == []


def test_issues_created_before_a_failure_keep_their_keys(client_as, store, jira):
    alex = client_as(ALEX)
    connect(alex)
    meeting = create(alex)
    first, second = task(meeting["id"], 1), task(meeting["id"], 2)
    processed(store, meeting, first, second)
    jira.crash_on_create = 2

    response = push(alex, meeting["id"], first.id, second.id)

    assert response.status_code == 200, response.text
    assert [r["key"] for r in response.json()] == ["DS-1", None]
    tasks = alex.get(f"/meetings/{meeting['id']}/report").json()["tasks"]
    assert [t["key"] for t in tasks] == ["DS-1", None]
    assert alex.get(f"/meetings/{meeting['id']}").json()["status"] == "needs_review"

    again = push(alex, meeting["id"], first.id, second.id)  # the retry creates only the second

    assert [r["key"] for r in again.json()] == ["DS-1", "DS-2"]
    assert [fields["summary"] for fields in jira.created] == ["Task 1", "Task 2"]


@pytest.mark.anyio
async def test_two_pushes_at_once_create_each_issue_once(app, client_as, store, jira):
    alex = client_as(ALEX)  # registers Alex as a test user
    meeting, draft = await anyio.to_thread.run_sync(prepared_meeting, alex, store)
    jira.slow = 0.05
    body = {"task_ids": [draft.id], "destination": "jira", "approved_by": "x"}
    results: list[httpx.Response] = []

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://brain.test",
        headers={"X-Test-User": ALEX.id},
    ) as client:

        async def push_once() -> None:
            results.append(await client.post(f"/meetings/{meeting['id']}/tasks/push", json=body))

        async with anyio.create_task_group() as group:
            group.start_soon(push_once)
            group.start_soon(push_once)

    assert [r.status_code for r in results] == [200, 200]
    assert [r.json()[0]["key"] for r in results] == ["DS-1", "DS-1"]
    assert len(jira.created) == 1


def prepared_meeting(alex: TestClient, store):
    connect(alex)
    meeting = create(alex)
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)
    return meeting, draft
