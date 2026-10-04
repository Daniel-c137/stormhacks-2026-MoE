"""An admin connects the team's Jira account in Settings; approved task drafts then become real
issues on that site, pushed by an admin. Alex is the team's admin; Sarah is a member."""

import asyncio

import pytest
from api_support import ALEX, AUTH_SECRET, OUTSIDER, SARAH, TEAM, create
from fastapi.testclient import TestClient
from jira_cloud_support import EMAIL, SITE, TOKEN, FakeJiraCloud, text_of
from test_report_review_api import processed, push, task

from brain.api.deps import get_http_transport, get_settings
from brain.sealing import unseal

ADMINS_ONLY = "Only an admin can do this"
CONNECT = {"site": SITE, "email": EMAIL, "api_token": TOKEN, "project": "DS"}


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
        "site": SITE,
        "project": "DS",
        "connected": True,
        "account_email": EMAIL,
    }
    saved = account(store)
    assert (saved.site, saved.email, saved.connected_by) == (SITE, EMAIL, ALEX.id)
    assert [r.url.path for r in jira.requests] == ["/rest/api/3/myself", "/rest/api/3/project/DS"]


def test_the_token_is_kept_sealed_and_never_sent_back(client_as, store, jira):
    connected = connect(client_as(ALEX))
    seen_by_member = client_as(SARAH).get("/settings")

    saved = account(store)
    assert TOKEN not in saved.sealed_token
    assert unseal(saved.sealed_token, AUTH_SECRET) == TOKEN
    assert TOKEN not in connected.text and TOKEN not in seen_by_member.text
    assert seen_by_member.json()["jira"]["account_email"] == EMAIL


def test_a_pasted_site_link_and_a_lowercase_key_are_tidied(client_as, jira):
    response = connect(client_as(ALEX), site=f"https://{SITE}/jira/software", project="ds")

    assert response.status_code == 200, response.text
    assert (response.json()["jira"]["site"], response.json()["jira"]["project"]) == (SITE, "DS")


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
    assert response.json()["jira"] == {
        "site": SITE,
        "project": "DS",
        "connected": False,
        "account_email": None,
    }
    assert account(store) is None


def test_another_site_in_the_connectors_disconnects_the_account(client_as, store, jira):
    alex = client_as(ALEX)
    connect(alex)

    same_site = alex.put("/settings/connectors", json={"jira": {"site": SITE, "project": "OPS"}})
    assert same_site.json()["jira"]["connected"] is True
    assert account(store) is not None

    moved = alex.put(
        "/settings/connectors", json={"jira": {"site": "other.atlassian.net", "project": "OPS"}}
    )
    assert moved.json()["jira"] == {
        "site": "other.atlassian.net",
        "project": "OPS",
        "connected": False,
        "account_email": None,
    }
    assert account(store) is None


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
    report = alex.get(f"/meetings/{meeting['id']}/report").json()
    assert (report["tasks"][0]["key"], report["tasks"][0]["jira_status"]) == ("DS-1", "todo")
    assert alex.get(f"/meetings/{meeting['id']}").json()["status"] == "pushed"


def test_the_push_goes_to_the_project_chosen_in_settings(client_as, store, jira):
    jira.projects.add("OPS")
    alex = client_as(ALEX)
    connect(alex)
    alex.put("/settings/connectors", json={"jira": {"site": SITE, "project": "OPS"}})
    meeting = create(alex)
    draft = task(meeting["id"], 1)
    processed(store, meeting, draft)

    response = push(alex, meeting["id"], draft.id)

    assert response.json()[0]["key"] == "OPS-1"


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


# the connector status


def test_a_connected_account_shows_jira_as_connected_and_says_what_is_missing(client_as, jira):
    alex = client_as(ALEX)
    before = {c["name"]: c for c in alex.get("/settings/connectors").json()}
    connect(alex)

    after = {c["name"]: c for c in alex.get("/settings/connectors").json()}

    assert before["jira"]["state"] == "not_configured"
    assert after["jira"]["state"] == "connected"
    assert EMAIL in after["jira"]["detail"] and "JIRA_MCP_URL" in after["jira"]["detail"]
