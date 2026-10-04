"""Final transcript segments: the worker saves them, team members read them back."""

import asyncio
import json
from datetime import UTC, datetime

import pytest
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


def started(client_as, *people) -> dict:
    """A live meeting hosted by Alex that these people (Alex and Sarah by default) joined
    through the link, as everyone does."""
    meeting = create(client_as(ALEX))
    for person in people or (ALEX, SARAH):
        assert client_as(person).post(f"/meetings/join/{meeting['code']}").status_code == 200
    return meeting


def ingest(worker: TestClient, meeting_id: str, *segments: dict):
    # Raw JSON so NaN and Infinity reach the server the way a buggy worker could send them.
    return worker.post(
        f"/internal/meetings/{meeting_id}/segments",
        content=json.dumps({"segments": list(segments)}),
        headers={"Content-Type": "application/json"},
    )


def transcript(client: TestClient, meeting_id: str):
    return client.get(f"/meetings/{meeting_id}/transcript")


# who may post


def test_request_without_the_worker_token_is_refused(app, client_as):
    meeting = started(client_as)

    response = TestClient(app).post(
        f"/internal/meetings/{meeting['id']}/segments",
        json={"segments": [segment(meeting["id"], 1)]},
    )

    assert response.status_code == 401
    assert transcript(client_as(ALEX), meeting["id"]).json() == []


def test_request_with_a_wrong_worker_token_is_refused(app, client_as):
    meeting = started(client_as)
    impostor = TestClient(app, headers={"X-Internal-Token": "guess"})

    response = ingest(impostor, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 401


def test_internal_routes_are_unavailable_when_no_worker_token_is_configured(app, client_as):
    meeting = started(client_as)
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    worker = TestClient(app, headers={"X-Internal-Token": WORKER_TOKEN})

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 503


def test_a_non_ascii_token_is_refused_not_an_error(app, client_as):
    meeting = started(client_as)
    odd = TestClient(app, headers={"X-Internal-Token": "café".encode("latin-1")})

    response = ingest(odd, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 401


def test_a_short_worker_token_counts_as_not_configured(app, client_as):
    meeting = started(client_as)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, brain_internal_token="short"
    )
    worker = TestClient(app, headers={"X-Internal-Token": "short"})

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 503
    assert "32" in response.json()["detail"]


# what is saved


def test_worker_saves_final_segments_and_teammates_read_them_back(worker, client_as):
    meeting = started(client_as)
    first = segment(meeting["id"], 1, speaker=ALEX)
    second = segment(meeting["id"], 2, speaker=SARAH)

    assert ingest(worker, meeting["id"], first, second).status_code == 204

    untranslated = {"language": None, "original_text": None}  # English speech (#106)
    assert transcript(client_as(SARAH), meeting["id"]).json() == [
        first | untranslated,
        second | untranslated,
    ]


def test_transcript_is_in_time_order_whatever_order_segments_arrive(worker, client_as):
    meeting = started(client_as)
    late = segment(meeting["id"], 1, speaker=SARAH, t=30.0)
    early = segment(meeting["id"], 2, speaker=ALEX, t=4.0)

    ingest(worker, meeting["id"], late)
    ingest(worker, meeting["id"], early)

    times = [s["t_start"] for s in transcript(client_as(ALEX), meeting["id"]).json()]
    assert times == [4.0, 30.0]


def test_a_resent_segment_is_saved_once(worker, client_as):
    meeting = started(client_as)
    seg = segment(meeting["id"], 1)

    ingest(worker, meeting["id"], seg)
    ingest(worker, meeting["id"], seg, segment(meeting["id"], 2))

    saved = transcript(client_as(ALEX), meeting["id"]).json()
    assert [s["seg_id"] for s in saved] == [seg["seg_id"], segment(meeting["id"], 2)["seg_id"]]


def test_partial_segments_are_rejected_and_nothing_from_the_batch_is_saved(worker, client_as):
    meeting = started(client_as)

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
    meeting, other = started(client_as), started(client_as)

    response = ingest(worker, meeting["id"], segment(other["id"], 1))

    assert response.status_code == 422
    assert transcript(alex, meeting["id"]).json() == []
    assert transcript(alex, other["id"]).json() == []


def test_segments_for_an_unknown_meeting_are_not_found(worker):
    response = ingest(worker, "no-such-meeting", segment("no-such-meeting", 1))

    assert response.status_code == 404


def test_last_words_after_the_host_ends_are_still_saved(worker, client_as):
    alex = client_as(ALEX)
    meeting = started(client_as)
    alex.post(f"/meetings/{meeting['id']}/end")

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 204
    assert len(transcript(alex, meeting["id"]).json()) == 1


# who may read


def test_another_team_cannot_read_the_transcript(worker, client_as):
    meeting = started(client_as)
    ingest(worker, meeting["id"], segment(meeting["id"], 1))

    response = transcript(client_as(OUTSIDER), meeting["id"])

    assert response.status_code == 404


# after retention deleted the transcript

DELETED_AT = datetime(2026, 10, 20, 3, 0, tzinfo=UTC)


def deleted_by_retention(store, worker, client_as) -> dict:
    alex = client_as(ALEX)
    meeting = started(client_as)
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


# what is refused, so the record everything reads from stays clean


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"text": "   "}, "blank text"),
        ({"text": "x" * 5001}, "text over 5000 characters"),
        ({"t_start": -1.0, "t_end": 1.0}, "negative start"),
        ({"t_start": 5.0, "t_end": 4.0}, "ends before it starts"),
        ({"t_start": float("nan")}, "NaN time"),
        ({"t_end": float("inf")}, "infinite time"),
        ({"speaker_id": "u-evil", "speaker_name": "Evil"}, "speaker not in the meeting"),
        ({"speaker_id": "Alex Chen"}, "display name instead of the account id"),
    ],
)
def test_a_bad_segment_rejects_the_batch(worker, client_as, change, why):
    meeting = started(client_as)

    response = ingest(
        worker, meeting["id"], segment(meeting["id"], 1), segment(meeting["id"], 2, **change)
    )

    assert response.status_code == 422, why
    assert transcript(client_as(ALEX), meeting["id"]).json() == []


def test_a_teammate_who_never_joined_is_not_a_speaker(worker, client_as):
    meeting = started(client_as, ALEX)  # Sarah is on the team but never joined

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1, speaker=SARAH))

    assert response.status_code == 422


def test_the_assistant_speaking_is_a_valid_speaker(worker, client_as):
    meeting = started(client_as)
    spoken = segment(meeting["id"], 1) | {"speaker_id": "agent", "speaker_name": "OmniMan"}

    assert ingest(worker, meeting["id"], spoken).status_code == 204


def test_a_batch_over_200_segments_is_rejected(worker, client_as):
    meeting = started(client_as)

    response = ingest(worker, meeting["id"], *(segment(meeting["id"], n) for n in range(201)))

    assert response.status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["needs_review", "pushed"])
async def test_segments_are_refused_once_the_report_exists(worker, client_as, store, status):
    meeting = started(client_as)
    await store.set_status(meeting["id"], status)

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1))

    assert response.status_code == 409


def test_a_reused_seg_id_with_different_content_is_a_conflict_not_silent(worker, client_as):
    meeting = started(client_as)
    first = segment(meeting["id"], 1)
    ingest(worker, meeting["id"], first)

    response = ingest(worker, meeting["id"], first | {"text": "Something else entirely"})

    assert response.status_code == 409
    assert [s["text"] for s in transcript(client_as(ALEX), meeting["id"]).json()] == [first["text"]]


def test_segments_starting_together_come_back_in_seg_id_order(worker, client_as):
    meeting = started(client_as)
    b = segment(meeting["id"], 2, t=7.0) | {"seg_id": "b", "t_end": 9.5}
    a = segment(meeting["id"], 1, t=7.0, speaker=SARAH) | {"seg_id": "a", "t_end": 9.5}

    ingest(worker, meeting["id"], b, a)

    assert [s["seg_id"] for s in transcript(client_as(ALEX), meeting["id"]).json()] == ["a", "b"]


# translated speech (#106)


def test_a_translated_segment_is_saved_with_its_language_and_original(worker, client_as):
    meeting = started(client_as)
    said = segment(meeting["id"], 1) | {
        "text": "We keep Postgres for now.",
        "language": "es",
        "original_text": "Por ahora nos quedamos con Postgres.",
    }

    assert ingest(worker, meeting["id"], said).status_code == 204

    [saved] = transcript(client_as(SARAH), meeting["id"]).json()
    assert (saved["text"], saved["language"], saved["original_text"]) == (
        "We keep Postgres for now.",
        "es",
        "Por ahora nos quedamos con Postgres.",
    )


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"language": "es", "original_text": "   "}, "blank original"),
        ({"language": "es", "original_text": "x" * 5001}, "original over 5000 characters"),
        ({"language": "Spanish"}, "language is not an ISO code"),
        ({"language": "ES"}, "language is not lowercase"),
    ],
)
def test_a_bad_translation_field_rejects_the_batch(worker, client_as, change, why):
    meeting = started(client_as)

    response = ingest(worker, meeting["id"], segment(meeting["id"], 1, **change))

    assert response.status_code == 422, why
    assert transcript(client_as(ALEX), meeting["id"]).json() == []
