"""Meeting lifecycle over HTTP: create, list, get, join by link, end."""

import jwt
import pytest
from api_support import (
    ALEX,
    KEY,
    LIVEKIT_URL,
    NOBODY,
    OUTSIDER,
    SARAH,
    SECRET,
    TEAM,
    FakeRooms,
    create,
)
from livekit.api import TokenVerifier

from brain.api.deps import app_settings, get_rooms
from brain.config import Settings

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


def test_title_is_capped_at_200_characters(client_as):
    alex = client_as(ALEX)

    assert alex.post("/meetings", json={"title": "x" * 200}).status_code == 200
    assert alex.post("/meetings", json={"title": "x" * 201}).status_code == 422


@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "/meetings"), ("get", "/meetings"), ("post", "/meetings/join/any-code")],
)
def test_someone_on_no_team_is_refused(client_as, method, path):
    meeting = create(client_as(ALEX))
    if "join" in path:
        path = f"/meetings/join/{meeting['code']}"

    response = client_as(NOBODY).request(method.upper(), path, json={"title": "Mine"})

    assert response.status_code == 403


# list and get


def test_list_shows_only_my_teams_meetings(client_as):
    mine = create(client_as(ALEX))
    create(client_as(OUTSIDER))

    listed = client_as(SARAH).get("/meetings").json()

    assert [m["id"] for m in listed] == [mine["id"]]


def test_list_is_newest_first(client_as):
    alex = client_as(ALEX)
    ids = [create(alex, f"Meeting {n}")["id"] for n in range(3)]

    listed = [m["id"] for m in alex.get("/meetings").json()]

    assert listed == list(reversed(ids))


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


def test_join_token_is_short_lived(client_as):
    meeting = create(client_as(ALEX))

    token = client_as(SARAH).post(f"/meetings/join/{meeting['code']}").json()["token"]

    claims = jwt.decode(token, SECRET, algorithms=["HS256"], options={"verify_aud": False})
    assert claims["exp"] - claims["nbf"] == 600


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
    app.dependency_overrides[app_settings] = lambda: Settings(_env_file=None)

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


def test_ending_records_how_long_the_meeting_ran(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)

    ended = alex.post(f"/meetings/{meeting['id']}/end").json()

    assert ended["duration_min"] is not None
    assert ended["duration_min"] >= 0


def test_ending_closes_the_livekit_room_once(client_as, rooms):
    alex = client_as(ALEX)
    meeting = create(alex)

    alex.post(f"/meetings/{meeting['id']}/end")
    alex.post(f"/meetings/{meeting['id']}/end")

    assert rooms.closed == [meeting["id"]]


def test_meeting_still_ends_when_livekit_cannot_close_the_room(app, client_as):
    app.dependency_overrides[get_rooms] = lambda: FakeRooms(fail=True)
    alex = client_as(ALEX)
    meeting = create(alex)

    response = alex.post(f"/meetings/{meeting['id']}/end")

    assert response.status_code == 200
    assert response.json()["status"] == "processing"


def test_another_team_cannot_end_the_meeting(client_as, rooms):
    meeting = create(client_as(ALEX))

    response = client_as(OUTSIDER).post(f"/meetings/{meeting['id']}/end")

    assert response.status_code == 404
    assert client_as(ALEX).get(f"/meetings/{meeting['id']}").json()["status"] == "live"
    assert rooms.closed == []
