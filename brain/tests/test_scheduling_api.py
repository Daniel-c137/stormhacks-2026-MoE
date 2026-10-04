"""Scheduled meetings, invitees and the workspace directory over HTTP."""

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM, create

from brain.store import InMemoryStore
from contracts import Person

PRIYA = Person(
    id="u-priya",
    name="Priya Natarajan",
    short="Priya",
    initials="PN",
    title="Payments engineer",
    email="priya@checkout.dev",
)
BEN = Person(
    id="u-ben",
    name="Ben Ortiz",
    short="Ben",
    initials="BO",
    title="Designer",
    email="ben@checkout.dev",
)

START = datetime(2026, 10, 5, 16, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def more_teammates(store: InMemoryStore) -> None:
    for person in (PRIYA, BEN):
        asyncio.run(store.upsert_person(person, TEAM.id))


def schedule(client, *, start=START, duration=30, invitees=(SARAH.id,), title="Retro") -> dict:
    response = client.post(
        "/meetings",
        json={
            "title": title,
            "scheduled_start": start.isoformat(),
            "duration_min": duration,
            "invitee_ids": list(invitees),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


# schedule


def test_a_start_time_creates_a_scheduled_meeting_with_its_invitees(client_as):
    meeting = schedule(client_as(ALEX), invitees=[SARAH.id, PRIYA.id])

    assert meeting["status"] == "scheduled"
    assert meeting["host_id"] == ALEX.id
    assert datetime.fromisoformat(meeting["scheduled_start"]) == START
    assert meeting["duration_min"] == 30
    assert meeting["invitee_ids"] == [SARAH.id, PRIYA.id]
    assert meeting["started_at"] is None


def test_without_a_start_time_the_meeting_is_live_now_with_its_invitees(client_as):
    response = client_as(ALEX).post(
        "/meetings", json={"title": "Huddle", "duration_min": 15, "invitee_ids": [BEN.id]}
    )

    meeting = response.json()
    assert meeting["status"] == "live"
    assert meeting["started_at"] is not None
    assert meeting["invitee_ids"] == [BEN.id]


def test_the_host_is_never_listed_as_an_invitee(client_as):
    meeting = schedule(client_as(ALEX), invitees=[ALEX.id, SARAH.id, SARAH.id])

    assert meeting["invitee_ids"] == [SARAH.id]


def test_invitees_outside_the_team_are_rejected_by_id(client_as):
    response = client_as(ALEX).post(
        "/meetings",
        json={
            "title": "Retro",
            "scheduled_start": START.isoformat(),
            "invitee_ids": [SARAH.id, OUTSIDER.id, "u-nobody"],
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert OUTSIDER.id in detail
    assert "u-nobody" in detail
    assert SARAH.id not in detail
    assert client_as(ALEX).get("/meetings").json() == []


@pytest.mark.parametrize("duration", [0, -15, 24 * 60 + 1])
def test_duration_must_be_positive_and_sane(client_as, duration):
    response = client_as(ALEX).post(
        "/meetings",
        json={"title": "Retro", "scheduled_start": START.isoformat(), "duration_min": duration},
    )

    assert response.status_code == 422


def test_start_time_without_a_timezone_is_rejected(client_as):
    response = client_as(ALEX).post(
        "/meetings", json={"title": "Retro", "scheduled_start": "2026-10-05T16:30:00"}
    )

    assert response.status_code == 422
    assert "timezone" in response.json()["detail"]


def test_start_time_in_another_timezone_keeps_the_instant(client_as):
    pacific = START.astimezone(timezone(timedelta(hours=-7)))

    meeting = schedule(client_as(ALEX), start=pacific)

    assert datetime.fromisoformat(meeting["scheduled_start"]) == START


# join a scheduled meeting


def test_the_host_joining_starts_the_scheduled_meeting(client_as):
    alex = client_as(ALEX)
    meeting = schedule(alex)

    response = alex.post(f"/meetings/join/{meeting['code']}")

    assert response.status_code == 200, response.text
    joined = response.json()["meeting"]
    assert joined["status"] == "live"
    assert joined["started_at"] is not None
    assert joined["participant_ids"] == [ALEX.id]
    assert alex.get(f"/meetings/{meeting['id']}").json()["status"] == "live"


@pytest.mark.parametrize("person", [SARAH, BEN], ids=["invitee", "teammate"])
def test_others_joining_before_the_host_starts_it_are_told_when_it_starts(client_as, person):
    meeting = schedule(client_as(ALEX), invitees=[SARAH.id])

    response = client_as(person).post(f"/meetings/join/{meeting['code']}")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "not started yet" in detail
    assert START.isoformat() in detail
    after = client_as(ALEX).get(f"/meetings/{meeting['id']}").json()
    assert after["status"] == "scheduled"
    assert after["participant_ids"] == []


def test_invitees_and_teammates_join_once_the_host_has_started_it(client_as):
    alex = client_as(ALEX)
    meeting = schedule(alex, invitees=[SARAH.id])
    alex.post(f"/meetings/join/{meeting['code']}")

    for person in (SARAH, BEN):
        response = client_as(person).post(f"/meetings/join/{meeting['code']}")
        assert response.status_code == 200, response.text

    assert client_as(ALEX).get(f"/meetings/{meeting['id']}").json()["participant_ids"] == [
        ALEX.id,
        SARAH.id,
        BEN.id,
    ]


def test_outsider_cannot_join_a_scheduled_meeting(client_as):
    meeting = schedule(client_as(ALEX))

    response = client_as(OUTSIDER).post(f"/meetings/join/{meeting['code']}")

    assert response.status_code == 403


def test_an_ended_scheduled_meeting_cannot_be_joined_even_by_the_host(client_as):
    alex = client_as(ALEX)
    meeting = schedule(alex)
    alex.post(f"/meetings/join/{meeting['code']}")
    alex.post(f"/meetings/{meeting['id']}/end")

    response = alex.post(f"/meetings/join/{meeting['code']}")

    assert response.status_code == 409
    assert "ended" in response.json()["detail"]


# end


def test_ending_records_when_the_meeting_ended(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)

    ended = alex.post(f"/meetings/{meeting['id']}/end").json()

    assert ended["status"] == "processing"
    assert ended["ended_at"] is not None


def test_ending_twice_keeps_the_first_end_time(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    first = alex.post(f"/meetings/{meeting['id']}/end").json()

    second = alex.post(f"/meetings/{meeting['id']}/end")

    assert second.status_code == 200
    assert second.json()["ended_at"] == first["ended_at"]


@pytest.mark.anyio
async def test_of_overlapping_ends_only_one_does_the_transition():
    from brain.api.meetings import end_live_meeting

    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    meeting = await store.create_meeting(TEAM.id, "Standup", ALEX.id)

    results = await asyncio.gather(*(end_live_meeting(store, meeting.id) for _ in range(5)))

    assert [ended for _, ended in results].count(True) == 1
    assert {m.status for m, _ in results} == {"processing"}


@pytest.mark.anyio
async def test_ending_a_meeting_that_is_not_live_does_not_transition():
    from brain.api.meetings import end_live_meeting

    store = InMemoryStore(teams=[TEAM], people=[ALEX])
    meeting = await store.create_meeting(TEAM.id, "Later", ALEX.id, scheduled_start=START)

    current, ended = await end_live_meeting(store, meeting.id)

    assert ended is False
    assert current.status == "scheduled"


# list


def test_list_has_scheduled_live_and_past_meetings_in_store_order(client_as, store):
    alex = client_as(ALEX)
    past = create(alex, "Past")
    alex.post(f"/meetings/{past['id']}/end")
    live = create(alex, "Live")
    later = schedule(alex, start=datetime.now(UTC) + timedelta(days=2), title="Later")
    create(client_as(OUTSIDER), "Theirs")

    listed = client_as(SARAH).get("/meetings").json()

    expected = asyncio.run(store.meetings(TEAM.id))
    assert [m["id"] for m in listed] == [m.id for m in expected]
    assert {m["id"]: m["status"] for m in listed} == {
        past["id"]: "processing",
        live["id"]: "live",
        later["id"]: "scheduled",
    }
    scheduled = next(m for m in listed if m["id"] == later["id"])
    assert scheduled["invitee_ids"] == [SARAH.id]
    assert scheduled["duration_min"] == 30


# invitees


def test_host_invites_teammates_to_a_meeting(client_as):
    alex = client_as(ALEX)
    meeting = schedule(alex, invitees=[SARAH.id])

    response = alex.post(
        f"/meetings/{meeting['id']}/invitees", json={"person_ids": [PRIYA.id, SARAH.id, ALEX.id]}
    )

    assert response.status_code == 200, response.text
    assert response.json()["invitee_ids"] == [SARAH.id, PRIYA.id]


def test_host_can_invite_during_a_live_meeting(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)

    response = alex.post(f"/meetings/{meeting['id']}/invitees", json={"person_ids": [BEN.id]})

    assert response.json()["invitee_ids"] == [BEN.id]


def test_inviting_someone_outside_the_team_names_them(client_as):
    alex = client_as(ALEX)
    meeting = schedule(alex)

    response = alex.post(
        f"/meetings/{meeting['id']}/invitees", json={"person_ids": [BEN.id, OUTSIDER.id]}
    )

    assert response.status_code == 422
    assert OUTSIDER.id in response.json()["detail"]
    assert alex.get(f"/meetings/{meeting['id']}").json()["invitee_ids"] == [SARAH.id]


def test_host_removes_an_invitee(client_as):
    alex = client_as(ALEX)
    meeting = schedule(alex, invitees=[SARAH.id, PRIYA.id])

    response = alex.delete(f"/meetings/{meeting['id']}/invitees/{SARAH.id}")

    assert response.status_code == 200, response.text
    assert response.json()["invitee_ids"] == [PRIYA.id]


def test_removing_someone_not_invited_is_harmless(client_as):
    alex = client_as(ALEX)
    meeting = schedule(alex, invitees=[SARAH.id])

    response = alex.delete(f"/meetings/{meeting['id']}/invitees/{BEN.id}")

    assert response.status_code == 200
    assert response.json()["invitee_ids"] == [SARAH.id]


def test_only_the_host_changes_invitees(client_as):
    meeting = schedule(client_as(ALEX), invitees=[SARAH.id])
    sarah = client_as(SARAH)

    added = sarah.post(f"/meetings/{meeting['id']}/invitees", json={"person_ids": [BEN.id]})
    removed = sarah.delete(f"/meetings/{meeting['id']}/invitees/{SARAH.id}")

    assert added.status_code == 403
    assert removed.status_code == 403
    assert client_as(ALEX).get(f"/meetings/{meeting['id']}").json()["invitee_ids"] == [SARAH.id]


def test_another_teams_meeting_invitees_look_like_they_do_not_exist(client_as):
    meeting = schedule(client_as(ALEX))
    olga = client_as(OUTSIDER)

    added = olga.post(f"/meetings/{meeting['id']}/invitees", json={"person_ids": [OUTSIDER.id]})
    removed = olga.delete(f"/meetings/{meeting['id']}/invitees/{SARAH.id}")

    assert added.status_code == 404
    assert removed.status_code == 404


def test_ended_meeting_invitees_cannot_change(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    alex.post(f"/meetings/{meeting['id']}/end")

    response = alex.post(f"/meetings/{meeting['id']}/invitees", json={"person_ids": [BEN.id]})

    assert response.status_code == 409


# directory


def test_directory_lists_my_workspace_by_name(client_as):
    response = client_as(SARAH).get("/directory")

    assert response.status_code == 200, response.text
    assert [p["id"] for p in response.json()] == [ALEX.id, BEN.id, PRIYA.id, SARAH.id]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("priya", [PRIYA.id]),
        ("BEN@CHECKOUT", [BEN.id]),
        ("designer", [BEN.id]),
        ("kim", [SARAH.id]),
    ],
)
def test_directory_searches_name_email_and_title(client_as, query, expected):
    response = client_as(ALEX).get("/directory", params={"q": query})

    assert [p["id"] for p in response.json()] == expected


def test_directory_never_shows_other_workspaces(client_as):
    assert client_as(ALEX).get("/directory", params={"q": "olga"}).json() == []
    assert [p["id"] for p in client_as(OUTSIDER).get("/directory").json()] == [OUTSIDER.id]


def test_directory_limit_caps_the_results(client_as):
    response = client_as(ALEX).get("/directory", params={"limit": 2})

    assert [p["id"] for p in response.json()] == [ALEX.id, BEN.id]


@pytest.mark.parametrize("limit", [0, 101])
def test_directory_limit_must_be_sane(client_as, limit):
    assert client_as(ALEX).get("/directory", params={"limit": limit}).status_code == 422
