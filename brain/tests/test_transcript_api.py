"""Final transcript segments: the worker saves them, team members read them back."""

import asyncio
from datetime import UTC, datetime

from api_support import ALEX, OUTSIDER, SARAH, WORKER_TOKEN, create
from fastapi.testclient import TestClient

from brain.api.deps import get_settings
from brain.config import Settings


def segment(meeting_id: str, n: int, *, speaker=ALEX, t: float | None = None, **overrides):
    t_start = float(n * 5) if t is None else t
    return {
        "seg_id": f"{meeting_id}-{speaker.id}-{n}",
        "meeting_id": meeting_id,
        "speaker_id": speaker.id,
        "speaker_name": speaker.name,
        "text": f"{speaker.short} says thing {n}",
        "is_final": True,
        "t_start": t_start,
        "t_end": t_start + 2.5,
    } | overrides


def ingest(worker: TestClient, meeting_id: str, *segments: dict):
    return worker.post(
        f"/internal/meetings/{meeting_id}/segments", json={"segments": list(segments)}
    )


def transcript(client: TestClient, meeting_id: str):
    return client.get(f"/meetings/{meeting_id}/transcript")


# who may post


def test_request_without_the_worker_token_is_refused(app, client_as):
    meeting = create(client_as(ALEX))

    response = TestClient(app).post(
        f"/internal/meetings/{meeting['id']}/segments",
        json={"segments": [segment(meeting["id"], 1)]},
    )

    assert response.status_code == 401
    assert transcript(client_as(ALEX), meeting["id"]).json() == []


def test_request_with_a_wrong_worker_token_is_refused(app, client_as):
    meeting = create(client_as(ALEX))
    impostor = TestClient(app, headers={"X-Internal-Token": "guess"})

    response = ingest(impostor, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 401


def test_internal_routes_are_unavailable_when_no_worker_token_is_configured(app, client_as):
    meeting = create(client_as(ALEX))
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    worker = TestClient(app, headers={"X-Internal-Token": WORKER_TOKEN})

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 503


# what is saved


def test_worker_saves_final_segments_and_teammates_read_them_back(worker, client_as):
    meeting = create(client_as(ALEX))
    first = segment(meeting["id"], 1, speaker=ALEX)
    second = segment(meeting["id"], 2, speaker=SARAH)

    assert ingest(worker, meeting["id"], first, second).status_code == 204

    assert transcript(client_as(SARAH), meeting["id"]).json() == [first, second]


def test_transcript_is_in_time_order_whatever_order_segments_arrive(worker, client_as):
    meeting = create(client_as(ALEX))
    late = segment(meeting["id"], 1, speaker=SARAH, t=30.0)
    early = segment(meeting["id"], 2, speaker=ALEX, t=4.0)

    ingest(worker, meeting["id"], late)
    ingest(worker, meeting["id"], early)

    times = [s["t_start"] for s in transcript(client_as(ALEX), meeting["id"]).json()]
    assert times == [4.0, 30.0]


def test_a_resent_segment_is_saved_once(worker, client_as):
    meeting = create(client_as(ALEX))
    seg = segment(meeting["id"], 1)

    ingest(worker, meeting["id"], seg)
    ingest(worker, meeting["id"], seg, segment(meeting["id"], 2))

    saved = transcript(client_as(ALEX), meeting["id"]).json()
    assert [s["seg_id"] for s in saved] == [seg["seg_id"], segment(meeting["id"], 2)["seg_id"]]


def test_partial_segments_are_rejected_and_nothing_from_the_batch_is_saved(worker, client_as):
    meeting = create(client_as(ALEX))

    response = ingest(
        worker,
        meeting["id"],
        segment(meeting["id"], 1),
        segment(meeting["id"], 2, is_final=False),
    )

    assert response.status_code == 422
    assert transcript(client_as(ALEX), meeting["id"]).json() == []


def test_segments_for_another_meeting_are_rejected(worker, client_as):
    alex = client_as(ALEX)
    meeting, other = create(alex), create(alex)

    response = ingest(worker, meeting["id"], segment(other["id"], 1))

    assert response.status_code == 422
    assert transcript(alex, meeting["id"]).json() == []
    assert transcript(alex, other["id"]).json() == []


def test_segments_for_an_unknown_meeting_are_not_found(worker):
    response = ingest(worker, "no-such-meeting", segment("no-such-meeting", 1))

    assert response.status_code == 404


def test_last_words_after_the_host_ends_are_still_saved(worker, client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    alex.post(f"/meetings/{meeting['id']}/end")

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 204
    assert len(transcript(alex, meeting["id"]).json()) == 1


# who may read


def test_another_team_cannot_read_the_transcript(worker, client_as):
    meeting = create(client_as(ALEX))
    ingest(worker, meeting["id"], segment(meeting["id"], 1))

    response = transcript(client_as(OUTSIDER), meeting["id"])

    assert response.status_code == 404


# after retention deleted the transcript

DELETED_AT = datetime(2026, 10, 20, 3, 0, tzinfo=UTC)


def deleted_by_retention(store, worker, client_as) -> dict:
    alex = client_as(ALEX)
    meeting = create(alex)
    ingest(worker, meeting["id"], segment(meeting["id"], 1), segment(meeting["id"], 2))
    alex.post(f"/meetings/{meeting['id']}/end")
    asyncio.run(store.delete_transcript(meeting["id"], DELETED_AT))
    return meeting


def test_a_deleted_transcript_reads_as_empty_and_the_meeting_says_when(store, worker, client_as):
    meeting = deleted_by_retention(store, worker, client_as)
    sarah = client_as(SARAH)

    response = transcript(sarah, meeting["id"])

    assert response.status_code == 200
    assert response.json() == []
    deleted_at = sarah.get(f"/meetings/{meeting['id']}").json()["transcript_deleted_at"]
    assert datetime.fromisoformat(deleted_at) == DELETED_AT


def test_a_late_resend_cannot_bring_a_deleted_transcript_back(store, worker, client_as):
    meeting = deleted_by_retention(store, worker, client_as)

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1), segment(meeting["id"], 3))

    assert response.status_code == 409
    assert transcript(client_as(ALEX), meeting["id"]).json() == []
