"""Asking the agent over HTTP: the worker's invoke, a typed question in a meeting, and Home."""

import asyncio
import logging
from pathlib import Path

import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM, WORKER_TOKEN, create
from ask_support import citing, scripted
from fastapi.testclient import TestClient

from brain.agent.ask import PlannedCall
from brain.api.deps import get_llm, get_memory, get_settings
from brain.config import Settings
from brain.llm import MockEmbedder
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.report import TranscriptInput
from contracts import Answer, InvokeResponse, Source

FIXTURES = Path(__file__).parent / "fixtures"
STANDUP = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
WAITLIST = "What did we decide about the waitlist email?"
SEARCH = PlannedCall(tool="search_meetings", query="waitlist email")


@pytest.fixture
def rows() -> InMemoryMemoryStore:
    return InMemoryMemoryStore()


@pytest.fixture
def memory(app, rows) -> MeetingMemory:
    memory = MeetingMemory(MockEmbedder(), rows)
    app.dependency_overrides[get_memory] = lambda: memory
    return memory


@pytest.fixture
def llm(app):
    """The model's two calls: search meeting memory, then cite the waitlist decision."""
    llm = scripted(SEARCH, answer=citing("waitlist email", text="Hold it until v0.9.4."))
    app.dependency_overrides[get_llm] = lambda: llm
    return llm


def indexed_standup(client: TestClient, memory) -> dict:
    """A team meeting holding the standup's transcript, indexed for search."""
    meeting = create(client, "Friday standup")
    segments = [s.model_copy(update={"meeting_id": meeting["id"]}) for s in STANDUP.segments]
    asyncio.run(memory.index_meeting(TEAM.id, meeting["id"], segments))
    return meeting


def invocation(meeting_id: str, **changes) -> dict:
    return {
        "id": "inv-1",
        "meeting_id": meeting_id,
        "via": "voice",
        "visibility": "public",
        "asked_by_id": SARAH.id,
        "asked_by_name": SARAH.name,
        "question": WAITLIST,
    } | changes


def standup_source(meeting_id: str) -> dict:
    return Source(
        kind="meeting", label="Friday standup 00:24", meeting_id=meeting_id, t=24
    ).model_dump()


# the worker's invoke


def test_invoke_answers_with_cited_sources(client_as, worker, store, memory, llm):
    meeting = indexed_standup(client_as(ALEX), memory)
    recent = {
        "seg_id": "live-1",
        "meeting_id": meeting["id"],
        "speaker_id": SARAH.id,
        "speaker_name": SARAH.name,
        "text": "Wait, are we still sending the waitlist email today?",
        "is_final": True,
        "t_start": 61,
        "t_end": 64,
    }

    response = worker.post(
        f"/internal/meetings/{meeting['id']}/invoke",
        json={"invocation": invocation(meeting["id"]), "recent_segments": [recent]},
    )

    assert response.status_code == 200, response.text
    answer = InvokeResponse.model_validate(response.json()).answer
    assert answer.invocation_id == "inv-1"
    assert answer.text == "Hold it until v0.9.4."
    assert [s.model_dump() for s in answer.sources] == [standup_source(meeting["id"])]
    assert "are we still sending the waitlist email today?" in llm.calls[0].prompt
    assert SARAH.name in llm.calls[0].prompt


def test_invoke_needs_the_internal_token(app, client_as, store, memory, llm):
    meeting = create(client_as(ALEX))
    body = {"invocation": invocation(meeting["id"])}

    missing = TestClient(app).post(f"/internal/meetings/{meeting['id']}/invoke", json=body)
    wrong = TestClient(app, headers={"X-Internal-Token": WORKER_TOKEN + "x"}).post(
        f"/internal/meetings/{meeting['id']}/invoke", json=body
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert llm.calls == []


def test_invoke_needs_an_existing_meeting_that_matches_the_invocation(
    client_as, worker, memory, llm
):
    meeting = create(client_as(ALEX))
    other = create(client_as(ALEX), "Other")

    unknown = worker.post("/internal/meetings/nope/invoke", json={"invocation": invocation("nope")})
    mismatched = worker.post(
        f"/internal/meetings/{meeting['id']}/invoke",
        json={"invocation": invocation(other["id"])},
    )

    assert unknown.status_code == 404
    assert mismatched.status_code == 422
    assert llm.calls == []


def test_invoke_only_sees_recent_segments_of_its_own_meeting(client_as, worker, memory, llm):
    meeting = indexed_standup(client_as(ALEX), memory)
    stray = {
        "seg_id": "x-1",
        "meeting_id": "another-meeting",
        "speaker_id": SARAH.id,
        "speaker_name": SARAH.name,
        "text": "Something said in another room.",
        "is_final": True,
        "t_start": 1,
        "t_end": 2,
    }

    response = worker.post(
        f"/internal/meetings/{meeting['id']}/invoke",
        json={"invocation": invocation(meeting["id"]), "recent_segments": [stray]},
    )

    assert response.status_code == 422
    assert llm.calls == []


# a typed question in a meeting


def test_ask_in_a_meeting_is_asked_as_the_signed_in_user(client_as, store, memory, llm):
    meeting = indexed_standup(client_as(ALEX), memory)

    response = client_as(SARAH).post(
        f"/meetings/{meeting['id']}/ask", json={"question": WAITLIST, "visibility": "public"}
    )

    assert response.status_code == 200, response.text
    answer = Answer.model_validate(response.json())
    assert [s.model_dump() for s in answer.sources] == [standup_source(meeting["id"])]
    assert SARAH.id in llm.calls[0].prompt and SARAH.name in llm.calls[0].prompt


def test_ask_in_a_meeting_sees_its_recent_transcript(client_as, worker, store, memory, llm):
    meeting = indexed_standup(client_as(ALEX), memory)
    said = {
        "seg_id": "live-9",
        "meeting_id": meeting["id"],
        "speaker_id": ALEX.id,
        "speaker_name": ALEX.name,
        "text": "Marketing wants the waitlist email out on Friday.",
        "is_final": True,
        "t_start": 300,
        "t_end": 305,
    }
    assert (
        worker.post(f"/internal/meetings/{meeting['id']}/segments", json={"segments": [said]})
    ).status_code == 204

    client_as(SARAH).post(
        f"/meetings/{meeting['id']}/ask", json={"question": WAITLIST, "visibility": "public"}
    )

    assert "Marketing wants the waitlist email out on Friday." in llm.calls[0].prompt


def test_ask_in_another_teams_meeting_is_not_found(client_as, memory, llm):
    meeting = create(client_as(ALEX))

    response = client_as(OUTSIDER).post(
        f"/meetings/{meeting['id']}/ask", json={"question": WAITLIST, "visibility": "public"}
    )

    assert response.status_code == 404
    assert llm.calls == []


def test_a_private_question_and_its_answer_are_never_stored_or_logged(
    client_as, store, memory, rows, llm, caplog
):
    meeting = indexed_standup(client_as(ALEX), memory)
    secret = "Is it just me, or is the waitlist email plan confusing?"
    before = (
        dict(rows._rows),
        asyncio.run(store.meetings(TEAM.id)),
        asyncio.run(store.transcript(meeting["id"])),
    )

    with caplog.at_level(logging.DEBUG):
        response = client_as(SARAH).post(
            f"/meetings/{meeting['id']}/ask", json={"question": secret, "visibility": "private"}
        )

    assert response.status_code == 200, response.text
    after = (
        dict(rows._rows),
        asyncio.run(store.meetings(TEAM.id)),
        asyncio.run(store.transcript(meeting["id"])),
    )
    assert after == before
    assert asyncio.run(store.public_chat(meeting["id"])) == []
    assert secret not in caplog.text
    assert "Hold it until v0.9.4." not in caplog.text


# Home


def test_home_ask_answers_across_the_team_with_follow_up_history(client_as, store, memory, llm):
    meeting = indexed_standup(client_as(ALEX), memory)

    response = client_as(ALEX).post(
        "/ask",
        json={
            "question": "And what did we decide about it?",
            "visibility": "private",
            "history": [
                {"role": "user", "text": "Who found the email exploit?"},
                {"role": "agent", "text": "Carol, at the Friday standup."},
            ],
        },
    )

    assert response.status_code == 200, response.text
    answer = Answer.model_validate(response.json())
    assert [s.model_dump() for s in answer.sources] == [standup_source(meeting["id"])]
    for call in llm.calls:
        assert "Who found the email exploit?" in call.prompt
        assert "Carol, at the Friday standup." in call.prompt


def test_home_ask_never_reaches_another_teams_memory(client_as, store, memory, llm):
    indexed_standup(client_as(ALEX), memory)

    response = client_as(OUTSIDER).post(
        "/ask", json={"question": WAITLIST, "visibility": "private"}
    )

    assert response.status_code == 200, response.text
    answer = Answer.model_validate(response.json())
    assert answer.sources == []
    assert len(llm.calls) == 1  # only the plan; no evidence, so nothing to answer from


def test_without_a_database_meeting_memory_is_unavailable(client_as, llm):
    response = client_as(ALEX).post("/ask", json={"question": WAITLIST, "visibility": "public"})

    assert response.status_code == 200, response.text
    answer = Answer.model_validate(response.json())
    assert any("meeting memory" in u.lower() for u in answer.unavailable)


def test_an_unconfigured_model_is_a_503(app, client_as, memory):
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)

    response = client_as(ALEX).post("/ask", json={"question": WAITLIST, "visibility": "public"})

    assert response.status_code == 503
    assert "Gemini is not configured" in response.json()["detail"]
