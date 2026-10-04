"""Who may act on the agent's shared answer card (Speak, Post in chat, Dismiss): the worker's
GET /internal/meetings/{id}/card-permission, asked before each action. With who_can_allow
"host" only the meeting's host or an admin may; with "everyone", any participant. Never the agent
itself, nor someone who is not a participant of the meeting. Settings and admin status are read on
every request, so a change during the meeting takes effect at once.

Sarah hosts these meetings; Alex is the team's admin; Priya is a member who is neither."""

import asyncio

import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM
from fastapi.testclient import TestClient

from contracts import AGENT_PARTICIPANT_ID, CardPermissionResponse, Person

PRIYA = Person(id="u-priya", name="Priya Natarajan", short="Priya", initials="PN")
DANA = Person(id="u-dana", name="Dana Levi", short="Dana", initials="DL")  # never joins


@pytest.fixture(autouse=True)
def people(store):
    for person in (PRIYA, DANA):
        asyncio.run(store.upsert_person(person, TEAM.id))


def allow(store, who_can_allow: str) -> None:
    async def save():
        saved = await store.settings(TEAM.id)
        await store.save_settings(saved.model_copy(update={"who_can_allow": who_can_allow}))

    asyncio.run(save())


@pytest.fixture
def meeting(store):
    """A live meeting Sarah hosts, which Sarah, Alex and Priya joined (and someone on another
    team, who should never be let in, but is refused even if they were)."""

    async def make():
        meeting = await store.create_meeting(TEAM.id, "Refund sync", SARAH.id)
        for person in (SARAH, ALEX, PRIYA, OUTSIDER):
            meeting = await store.add_participant(meeting.id, person.id)
        return meeting

    return asyncio.run(make())


def allowed(worker: TestClient, meeting_id: str, participant_id: str) -> bool:
    response = worker.get(
        f"/internal/meetings/{meeting_id}/card-permission",
        params={"participant_id": participant_id},
    )
    assert response.status_code == 200, response.text
    return CardPermissionResponse.model_validate(response.json()).allowed


@pytest.mark.parametrize(
    ("who", "expected"),
    [
        (SARAH.id, True),  # the host
        (ALEX.id, True),  # an admin
        (PRIYA.id, False),  # a member who is neither
        (DANA.id, False),  # a member who is not a participant
        ("u-ghost", False),  # nobody's account
        (OUTSIDER.id, False),  # on another team
        (AGENT_PARTICIPANT_ID, False),  # the agent itself
    ],
)
def test_host_only_lets_the_host_or_an_admin_act(store, worker, meeting, who, expected):
    allow(store, "host")

    assert allowed(worker, meeting.id, who) is expected


@pytest.mark.parametrize(
    ("who", "expected"),
    [
        (SARAH.id, True),
        (ALEX.id, True),
        (PRIYA.id, True),  # any participant
        (DANA.id, False),
        ("u-ghost", False),
        (OUTSIDER.id, False),
        (AGENT_PARTICIPANT_ID, False),
    ],
)
def test_everyone_lets_any_participant_act(store, worker, meeting, who, expected):
    allow(store, "everyone")

    assert allowed(worker, meeting.id, who) is expected


def test_everyone_is_the_default(worker, meeting):
    assert allowed(worker, meeting.id, PRIYA.id) is True


def test_a_settings_change_during_the_meeting_takes_effect_at_once(store, worker, meeting):
    allow(store, "host")
    assert allowed(worker, meeting.id, PRIYA.id) is False

    allow(store, "everyone")
    assert allowed(worker, meeting.id, PRIYA.id) is True

    allow(store, "host")
    assert allowed(worker, meeting.id, PRIYA.id) is False


def test_granting_or_revoking_admin_during_the_meeting_takes_effect_at_once(store, worker, meeting):
    allow(store, "host")

    asyncio.run(store.update_person(PRIYA.model_copy(update={"is_admin": True})))
    assert allowed(worker, meeting.id, PRIYA.id) is True

    asyncio.run(store.update_person(ALEX.model_copy(update={"is_admin": False})))
    assert allowed(worker, meeting.id, ALEX.id) is False


def test_an_unknown_meeting_is_not_found(worker):
    response = worker.get(
        "/internal/meetings/nope/card-permission", params={"participant_id": SARAH.id}
    )

    assert response.status_code == 404


def test_the_participant_is_required(worker, meeting):
    response = worker.get(f"/internal/meetings/{meeting.id}/card-permission")

    assert response.status_code == 422


def test_only_the_worker_may_ask(app, meeting):
    response = TestClient(app).get(
        f"/internal/meetings/{meeting.id}/card-permission", params={"participant_id": SARAH.id}
    )

    assert response.status_code == 401
