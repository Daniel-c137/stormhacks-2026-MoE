"""What the realtime worker reads and saves while it sits in a meeting as the agent: the meeting
and the team's voice when it joins, and the meeting's public chat. Private chat never arrives."""

import asyncio
from datetime import UTC, datetime

from api_support import ALEX, OUTSIDER, SARAH, create
from fastapi.testclient import TestClient

from contracts import AGENT_PARTICIPANT_ID, WorkerMeetingResponse


def started(client_as, *people) -> dict:
    meeting = create(client_as(ALEX))
    for person in people or (ALEX, SARAH):
        assert client_as(person).post(f"/meetings/join/{meeting['code']}").status_code == 200
    return meeting


def message(meeting_id: str, n: int = 1, *, sender=ALEX, **overrides) -> dict:
    return {
        "id": f"lk-{n}",
        "meeting_id": meeting_id,
        "sender_id": sender.id,
        "sender_name": sender.name,
        "is_agent": False,
        "text": f"{sender.short} says {n}",
        "ts": f"2026-10-03T17:0{n}:00Z",
    } | overrides


def saved_chat(store, meeting_id: str) -> list:
    return asyncio.run(store.public_chat(meeting_id))


# the meeting the worker joined


def test_the_worker_reads_the_meeting_it_joined(worker, client_as):
    meeting = started(client_as)

    response = worker.get(f"/internal/meetings/{meeting['id']}")

    assert response.status_code == 200
    info = WorkerMeetingResponse.model_validate(response.json())
    assert info.meeting.id == meeting["id"]
    assert info.meeting.status == "live"
    assert info.meeting.started_at is not None  # the clock segment times are on
    assert info.voice_id is None  # the worker uses its default voice


def test_the_teams_chosen_voice_is_the_one_polaris_speaks_in(store, worker, client_as):
    meeting = started(client_as)
    settings = asyncio.run(store.settings(meeting["team_id"]))
    asyncio.run(store.save_settings(settings.model_copy(update={"voice": "v-team"})))

    info = WorkerMeetingResponse.model_validate(
        worker.get(f"/internal/meetings/{meeting['id']}").json()
    )

    assert info.voice_id == "v-team"


def test_a_room_that_is_not_a_meeting_is_not_found(worker):
    assert worker.get("/internal/meetings/not-a-meeting").status_code == 404


def test_reading_a_meeting_needs_the_worker_token(app, client_as):
    meeting = started(client_as)

    assert TestClient(app).get(f"/internal/meetings/{meeting['id']}").status_code == 401
    impostor = TestClient(app, headers={"X-Internal-Token": "guess"})
    assert impostor.get(f"/internal/meetings/{meeting['id']}").status_code == 401


# the agent joining


def joined_at(store, meeting_id: str):
    return asyncio.run(store.meeting(meeting_id)).agent_joined_at


def test_the_worker_records_that_polaris_joined_and_the_board_sees_it(store, worker, client_as):
    meeting = started(client_as)
    before = datetime.now(UTC)

    response = worker.post(f"/internal/meetings/{meeting['id']}/agent-joined")

    assert response.status_code == 204
    assert before <= joined_at(store, meeting["id"]) <= datetime.now(UTC)
    seen = client_as(ALEX).get(f"/meetings/{meeting['id']}").json()
    assert datetime.fromisoformat(seen["agent_joined_at"]) == joined_at(store, meeting["id"])


def test_a_meeting_polaris_never_joined_says_so(client_as):
    meeting = started(client_as)

    assert client_as(ALEX).get(f"/meetings/{meeting['id']}").json()["agent_joined_at"] is None


def test_joining_again_keeps_the_first_time(store, worker, client_as):
    meeting = started(client_as)
    worker.post(f"/internal/meetings/{meeting['id']}/agent-joined")
    first = joined_at(store, meeting["id"])

    again = worker.post(f"/internal/meetings/{meeting['id']}/agent-joined")

    assert again.status_code == 204
    assert joined_at(store, meeting["id"]) == first


def test_polaris_cannot_join_a_meeting_that_is_not_live(store, worker, client_as):
    meeting = started(client_as)
    client_as(ALEX).post(f"/meetings/{meeting['id']}/end")
    scheduled = asyncio.run(
        store.create_meeting(
            meeting["team_id"], "Later", ALEX.id, scheduled_start=datetime(2030, 1, 1, tzinfo=UTC)
        )
    )

    for meeting_id in (meeting["id"], scheduled.id):
        response = worker.post(f"/internal/meetings/{meeting_id}/agent-joined")
        assert response.status_code == 409
        assert joined_at(store, meeting_id) is None
    assert worker.post("/internal/meetings/not-a-meeting/agent-joined").status_code == 404


def test_recording_the_join_needs_the_worker_token(app, store, client_as):
    meeting = started(client_as)
    path = f"/internal/meetings/{meeting['id']}/agent-joined"

    assert TestClient(app).post(path).status_code == 401
    assert TestClient(app, headers={"X-Internal-Token": "guess"}).post(path).status_code == 401
    assert client_as(ALEX).post(path).status_code == 401
    assert joined_at(store, meeting["id"]) is None


# public chat


def test_public_chat_is_saved_once_per_message(store, worker, client_as):
    meeting = started(client_as)

    first = worker.post(f"/internal/meetings/{meeting['id']}/chat", json=message(meeting["id"]))
    again = worker.post(f"/internal/meetings/{meeting['id']}/chat", json=message(meeting["id"]))

    assert (first.status_code, again.status_code) == (204, 204)
    [saved] = saved_chat(store, meeting["id"])
    assert (saved.id, saved.sender_id, saved.text) == ("lk-1", ALEX.id, "Alex says 1")


def test_polaris_answers_in_chat_are_saved_too(store, worker, client_as):
    meeting = started(client_as)
    answer = message(
        meeting["id"],
        2,
        sender_id=AGENT_PARTICIPANT_ID,
        sender_name="Polaris",
        is_agent=True,
        text="The refund window is 14 days.",
    )

    assert worker.post(f"/internal/meetings/{meeting['id']}/chat", json=answer).status_code == 204
    assert [m.is_agent for m in saved_chat(store, meeting["id"])] == [True]


def test_private_chat_is_refused_and_never_stored(store, worker, client_as):
    meeting = started(client_as)
    private = message(meeting["id"], visibility="private", recipient_id=SARAH.id)

    response = worker.post(f"/internal/meetings/{meeting['id']}/chat", json=private)

    assert response.status_code == 422
    assert saved_chat(store, meeting["id"]) == []


def test_chat_must_belong_to_the_meeting_and_come_from_a_participant(store, worker, client_as):
    meeting = started(client_as)
    other = started(client_as)

    elsewhere = worker.post(f"/internal/meetings/{meeting['id']}/chat", json=message(other["id"]))
    stranger = worker.post(
        f"/internal/meetings/{meeting['id']}/chat", json=message(meeting["id"], sender=OUTSIDER)
    )
    blank = worker.post(
        f"/internal/meetings/{meeting['id']}/chat", json=message(meeting["id"], text="  ")
    )
    huge = worker.post(
        f"/internal/meetings/{meeting['id']}/chat", json=message(meeting["id"], text="x" * 5001)
    )

    assert [r.status_code for r in (elsewhere, stranger, blank, huge)] == [422] * 4
    assert saved_chat(store, meeting["id"]) == []


def test_chat_for_an_unknown_meeting_is_not_found(worker):
    response = worker.post("/internal/meetings/nope/chat", json=message("nope"))

    assert response.status_code == 404


def test_chat_is_taken_until_the_report_exists(store, worker, client_as):
    meeting = started(client_as)
    client_as(ALEX).post(f"/meetings/{meeting['id']}/end")  # processing: last messages land

    late = worker.post(f"/internal/meetings/{meeting['id']}/chat", json=message(meeting["id"]))
    asyncio.run(store.set_status(meeting["id"], "needs_review"))
    after = worker.post(f"/internal/meetings/{meeting['id']}/chat", json=message(meeting["id"], 2))

    assert (late.status_code, after.status_code) == (204, 409)
    assert [m.id for m in saved_chat(store, meeting["id"])] == ["lk-1"]


def test_saving_chat_needs_the_worker_token(app, client_as, store):
    meeting = started(client_as)

    response = TestClient(app).post(
        f"/internal/meetings/{meeting['id']}/chat", json=message(meeting["id"])
    )

    assert response.status_code == 401
    assert saved_chat(store, meeting["id"]) == []
