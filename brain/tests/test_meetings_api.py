"""Meeting lifecycle over HTTP: create, list, get, join by link, end.

Auth is overridden here; Supabase session resolution is its own slice. The store is the real
in-memory store and the LiveKit token is really signed, then verified with LiveKit's verifier.
"""

import pytest
from fastapi import Header
from fastapi.testclient import TestClient
from livekit.api import TokenVerifier

from brain.api.deps import current_user, get_settings, get_store
from brain.config import Settings
from brain.main import create_app
from brain.store import InMemoryStore
from contracts import Person, Team

KEY = "test-key"
SECRET = "test-secret-that-is-long-enough-for-hs256"
LIVEKIT_URL = "wss://omniroom-test.livekit.cloud"

ALEX = Person(id="u-alex", name="Alex Chen", short="Alex", initials="AC")
SARAH = Person(id="u-sarah", name="Sarah Kim", short="Sarah", initials="SK")
OUTSIDER = Person(id="u-olga", name="Olga Petrova", short="Olga", initials="OP")
NOBODY = Person(id="u-nobody", name="No Team", short="No", initials="NT")  # on no team
TEAM = Team(id="t-1", name="Checkout", member_ids=[ALEX.id, SARAH.id])
OTHER_TEAM = Team(id="t-2", name="Elsewhere", member_ids=[OUTSIDER.id])


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None, livekit_url=LIVEKIT_URL, livekit_api_key=KEY, livekit_api_secret=SECRET
    )


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(teams=[TEAM, OTHER_TEAM], people=[ALEX, SARAH, OUTSIDER])


PEOPLE = {p.id: p for p in [ALEX, SARAH, OUTSIDER, NOBODY]}


def user_from_test_header(x_test_user: str = Header()) -> Person:
    return PEOPLE[x_test_user]


@pytest.fixture
def app(store, settings):
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[current_user] = user_from_test_header
    return app


@pytest.fixture
def client_as(app):
    """client_as(person) -> a TestClient whose requests are made as that person. Each client
    carries its own identity, so several can be held at once."""

    def make(person: Person) -> TestClient:
        return TestClient(app, headers={"X-Test-User": person.id})

    return make


def create(client: TestClient, title: str = "Standup") -> dict:
    response = client.post("/meetings", json={"title": title})
    assert response.status_code == 200, response.text
    return response.json()


# test harness


def test_two_clients_held_at_once_keep_their_own_identity(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    client_as(OUTSIDER)

    assert alex.get(f"/meetings/{meeting['id']}").status_code == 200


# create


def test_member_creates_a_live_meeting_they_host_in_their_team(client_as):
    meeting = create(client_as(ALEX), "Sprint planning")

    assert meeting["title"] == "Sprint planning"
    assert meeting["status"] == "live"
    assert meeting["host_id"] == ALEX.id
    assert meeting["team_id"] == TEAM.id
    assert meeting["started_at"] is not None


def test_each_meeting_gets_its_own_url_safe_join_code(client_as):
    alex = client_as(ALEX)
    codes = {create(alex)["code"] for _ in range(20)}

    assert len(codes) == 20
    for code in codes:
        assert len(code) >= 8
        assert code.replace("-", "").replace("_", "").isalnum()


def test_blank_title_is_rejected(client_as):
    response = client_as(ALEX).post("/meetings", json={"title": "   "})

    assert response.status_code == 422


# list and get


def test_list_shows_only_my_teams_meetings(client_as):
    mine = create(client_as(ALEX))
    create(client_as(OUTSIDER))

    listed = client_as(SARAH).get("/meetings").json()

    assert [m["id"] for m in listed] == [mine["id"]]


def test_teammate_can_get_a_meeting(client_as):
    meeting = create(client_as(ALEX))

    response = client_as(SARAH).get(f"/meetings/{meeting['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == meeting["id"]


def test_another_teams_meeting_looks_like_it_does_not_exist(client_as):
    meeting = create(client_as(ALEX))

    response = client_as(OUTSIDER).get(f"/meetings/{meeting['id']}")

    assert response.status_code == 404


# join by link


def test_teammate_joins_by_code_and_gets_a_livekit_token_for_that_room(client_as):
    meeting = create(client_as(ALEX))

    response = client_as(SARAH).post(f"/meetings/join/{meeting['code']}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["livekit_url"] == LIVEKIT_URL
    assert body["meeting"]["id"] == meeting["id"]
    claims = TokenVerifier(KEY, SECRET).verify(body["token"])
    assert claims.identity == SARAH.id
    assert claims.name == SARAH.name
    assert claims.video.room == meeting["id"]
    assert not claims.video.room_admin


def test_host_joining_gets_room_admin(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)

    body = alex.post(f"/meetings/join/{meeting['code']}").json()

    assert TokenVerifier(KEY, SECRET).verify(body["token"]).video.room_admin is True


def test_joining_records_the_participant_once(client_as):
    meeting = create(client_as(ALEX))
    sarah = client_as(SARAH)

    sarah.post(f"/meetings/join/{meeting['code']}")
    body = sarah.post(f"/meetings/join/{meeting['code']}").json()

    assert body["meeting"]["participant_ids"] == [SARAH.id]


def test_non_member_with_the_link_is_refused_a_token(client_as):
    meeting = create(client_as(ALEX))

    response = client_as(OUTSIDER).post(f"/meetings/join/{meeting['code']}")

    assert response.status_code == 403
    assert "token" not in response.json()


def test_unknown_code_is_not_found(client_as):
    response = client_as(SARAH).post("/meetings/join/no-such-code")

    assert response.status_code == 404


def test_ended_meeting_cannot_be_joined(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    alex.post(f"/meetings/{meeting['id']}/end")

    response = client_as(SARAH).post(f"/meetings/join/{meeting['code']}")

    assert response.status_code == 409


def test_join_without_livekit_credentials_is_unavailable_not_faked(app, client_as):
    meeting = create(client_as(ALEX))
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)

    response = client_as(SARAH).post(f"/meetings/join/{meeting['code']}")

    assert response.status_code == 503
    assert "LiveKit" in response.json()["detail"]


# end


def test_host_ends_the_meeting_and_it_moves_to_processing(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)

    response = alex.post(f"/meetings/{meeting['id']}/end")

    assert response.status_code == 200
    assert response.json()["status"] == "processing"


def test_only_the_host_can_end_the_meeting(client_as):
    meeting = create(client_as(ALEX))

    response = client_as(SARAH).post(f"/meetings/{meeting['id']}/end")

    assert response.status_code == 403
    assert client_as(SARAH).get(f"/meetings/{meeting['id']}").json()["status"] == "live"


def test_ending_twice_is_harmless(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    alex.post(f"/meetings/{meeting['id']}/end")

    response = alex.post(f"/meetings/{meeting['id']}/end")

    assert response.status_code == 200
    assert response.json()["status"] == "processing"
