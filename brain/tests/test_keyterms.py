"""Keyterms that bias the worker's Scribe transcription toward the agent's name, the team and
its work, within Scribe Realtime's limits."""

import logging
from datetime import UTC, datetime

import anyio
import pytest
from api_support import ALEX, SARAH, TEAM
from fastapi.testclient import TestClient

from brain.api.deps import get_settings
from brain.keyterms import MAX_KEYTERM_CHARS, MAX_KEYTERMS, fit, select
from contracts import (
    Agenda,
    AgendaItem,
    GitHubSettings,
    Identity,
    JiraSettings,
    KeytermsResponse,
    Person,
    TeamSettings,
    get_identity,
)

AGENT = get_identity().agent_name


# selection


def test_terms_are_trimmed_and_shortened_at_a_word_boundary():
    assert fit("  Polaris \n") == "Polaris"
    assert fit("Refund   the\tusers") == "Refund the users"
    assert fit("Refund the double-charged users") == "Refund the"
    assert fit("a" * MAX_KEYTERM_CHARS) == "a" * MAX_KEYTERM_CHARS
    assert fit("Waitlist email, rollout plan") == "Waitlist email"


def test_a_term_with_no_word_that_fits_is_dropped():
    assert fit("Supercalifragilisticexpialidocious") is None
    assert fit("Supercalifragilisticexpialidocious plan") is None
    assert fit("   ") is None
    assert fit("") is None


def test_selection_keeps_the_first_of_each_term_ignoring_case():
    assert select(["Polaris", "Alex Chen", "polaris", " Alex Chen ", "Alex"]) == [
        "Polaris",
        "Alex Chen",
        "Alex",
    ]


def test_selection_stops_at_the_term_limit_and_every_term_fits():
    terms = select([f"Term {n}" for n in range(MAX_KEYTERMS + 20)])

    assert MAX_KEYTERMS == 50 and MAX_KEYTERM_CHARS == 20
    assert terms == [f"Term {n}" for n in range(MAX_KEYTERMS)]


def test_terms_that_do_not_fit_take_no_place():
    terms = select(["x" * 30, *[f"T{n}" for n in range(MAX_KEYTERMS)]])

    assert len(terms) == MAX_KEYTERMS
    assert terms[0] == "T0"


# the worker's endpoint


def keyterms(client: TestClient, meeting_id: str, **kwargs):
    return client.get(f"/internal/meetings/{meeting_id}/keyterms", **kwargs)


def setup(store, *, agenda: list[str] = (), members: list[Person] = (), **team):
    """TEAM's settings, any extra members, and a live meeting with this agenda."""

    async def go():
        await store.save_settings(
            TeamSettings(
                team_id=TEAM.id,
                github=GitHubSettings(repo="dropsubs/checkout-app"),
                jira=JiraSettings(project="DS"),
                **team,
            )
        )
        for person in members:
            await store.upsert_person(person, TEAM.id)
        meeting = await store.create_meeting(TEAM.id, "Refund sync", ALEX.id)
        if agenda:
            items = [AgendaItem(id=f"a{n}", title=t) for n, t in enumerate(agenda)]
            await store.save_agenda(
                Agenda(meeting_id=meeting.id, items=items, generated_at=datetime.now(UTC))
            )
        return meeting

    return anyio.run(go)


def jira_issue(key: str, category: str = "indeterminate") -> dict:
    return {
        "key": key,
        "fields": {"summary": key, "status": {"name": key, "statusCategory": {"key": category}}},
    }


@pytest.fixture
def jira(app, settings, jira_over_http):
    """Jira configured for the deployment and served by a FakeJira on localhost."""
    fake, url = jira_over_http
    configured = settings.model_copy(
        update={"jira_mcp_url": url, "jira_base_url": "https://dropsubs.atlassian.net"}
    )
    app.dependency_overrides[get_settings] = lambda: configured
    return fake


def test_terms_come_in_order_agent_team_settings_agenda_then_jira(worker, store, jira):
    jira.issues = [jira_issue("DS-117"), jira_issue("DS-104", category="done"), jira_issue("DS-9")]
    meeting = setup(
        store,
        wake_phrase="Hi Atlas",
        agenda=["Refund the double-charged users", "Rollout"],
    )

    response = keyterms(worker, meeting.id)

    assert response.status_code == 200, response.text
    assert KeytermsResponse.model_validate(response.json()).terms == [
        AGENT,
        "Atlas",
        "Alex Chen",
        "Alex",
        "Sarah Kim",
        "Sarah",
        "checkout-app",
        "DS",
        "Refund the",
        "Rollout",
        "DS-117",
        "DS-9",
    ]
    (search,) = jira.searches
    assert "DS" in search["jql"] and "done" in search["jql"].lower()
    assert search["maxResults"] == MAX_KEYTERMS - 10


def test_the_default_wake_phrase_is_the_agent_name_alone():
    assert get_identity().wake_phrase == AGENT
    assert Identity(product_name="Acme", agent_name="Nova").wake_phrase == "Nova"


@pytest.mark.parametrize("phrase", ["Hey Atlas", "hi, Atlas", "OK Atlas", "Hello hey Atlas"])
def test_a_greeting_before_a_custom_wake_phrase_takes_no_keyterm(worker, store, phrase):
    meeting = setup(store, wake_phrase=phrase)

    terms = keyterms(worker, meeting.id).json()["terms"]

    assert terms[:3] == [AGENT, "Atlas", "Alex Chen"]


def test_a_wake_phrase_that_is_only_a_greeting_adds_nothing(worker, store):
    meeting = setup(store, wake_phrase="Hey")

    terms = keyterms(worker, meeting.id).json()["terms"]

    assert terms[:2] == [AGENT, "Alex Chen"]


def test_the_agent_name_comes_from_the_identity_never_hard_coded(worker, store, monkeypatch):
    monkeypatch.setattr(
        "brain.keyterms.get_identity", lambda: Identity(product_name="Acme", agent_name="Nova")
    )
    meeting = setup(store)

    terms = keyterms(worker, meeting.id).json()["terms"]

    assert terms[0] == "Nova"
    assert AGENT not in terms


def test_repeats_are_dropped_and_the_limits_hold(worker, store, jira):
    jira.issues = [jira_issue("DS-1")]
    crowd = [
        Person(id=f"u-{n}", name=f"Person{n} Surname{n}", short=f"P{n}", initials="PS")
        for n in range(30)
    ]
    meeting = setup(
        store,
        wake_phrase=f"Hey {AGENT}",
        members=crowd,
        agenda=["Alex Chen", "A topic far too long to fit in one keyterm"],
    )

    terms = keyterms(worker, meeting.id).json()["terms"]

    assert len(terms) == MAX_KEYTERMS
    assert all(0 < len(t) <= MAX_KEYTERM_CHARS for t in terms)
    assert len({t.casefold() for t in terms}) == len(terms)
    assert terms[:3] == [AGENT, "Alex Chen", "Alex"]
    assert jira.searches == []  # no room left, so Jira is not asked


def test_without_jira_configured_there_are_no_issue_keys(worker, store):
    meeting = setup(store, agenda=["Rollout"])

    response = keyterms(worker, meeting.id)

    assert response.status_code == 200
    assert response.json()["terms"][-1] == "Rollout"


def test_a_failing_jira_is_skipped_and_logged(worker, store, jira, caplog):
    jira.search_error = "Jira is having a bad day"
    meeting = setup(store, agenda=["Rollout"])

    with caplog.at_level(logging.WARNING, logger="brain.keyterms"):
        response = keyterms(worker, meeting.id)

    assert response.status_code == 200, response.text
    assert response.json()["terms"][-1] == "Rollout"
    assert "Jira" in caplog.text and "bad day" in caplog.text


def test_an_unreachable_jira_is_skipped(app, worker, store, settings):
    unreachable = settings.model_copy(
        update={
            "jira_mcp_url": "http://127.0.0.1:9/mcp",
            "jira_base_url": "https://dropsubs.atlassian.net",
        }
    )
    app.dependency_overrides[get_settings] = lambda: unreachable
    meeting = setup(store)

    response = keyterms(worker, meeting.id)

    assert response.status_code == 200, response.text
    assert response.json()["terms"][-1] == "DS"


def test_settings_left_blank_add_nothing(worker, store):
    meeting = setup(store)
    anyio.run(
        store.save_settings,
        TeamSettings(team_id=TEAM.id, github=GitHubSettings(), jira=JiraSettings()),
    )

    terms = keyterms(worker, meeting.id).json()["terms"]

    assert terms == [AGENT, "Alex Chen", "Alex", "Sarah Kim", "Sarah"]


def test_a_team_without_its_own_repo_or_project_uses_the_deployments(app, worker, store, settings):
    deployment = settings.model_copy(
        update={"github_repo": "dropsubs/payments", "jira_project_key": "PAY"}
    )
    app.dependency_overrides[get_settings] = lambda: deployment
    meeting = setup(store)
    anyio.run(
        store.save_settings,
        TeamSettings(team_id=TEAM.id, github=GitHubSettings(), jira=JiraSettings()),
    )

    terms = keyterms(worker, meeting.id).json()["terms"]

    assert terms[-2:] == ["payments", "PAY"]


def test_an_unknown_meeting_is_a_404(worker):
    assert keyterms(worker, "no-such-meeting").status_code == 404


def test_keyterms_need_the_internal_token(app, store):
    meeting = setup(store)
    anonymous = TestClient(app)

    assert keyterms(anonymous, meeting.id).status_code == 401
    assert keyterms(anonymous, meeting.id, headers={"X-Internal-Token": "wrong"}).status_code == 401
    assert (
        keyterms(TestClient(app, headers={"X-Test-User": SARAH.id}), meeting.id).status_code == 401
    )
