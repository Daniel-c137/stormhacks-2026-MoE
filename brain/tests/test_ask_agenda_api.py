"""#93 over HTTP: the meeting ask and the worker's invoke both see the asker's own agenda; Home
never does. The store is the in-memory one, or Postgres with BRAIN_TEST_STORE=postgres."""

import asyncio

import pytest
from api_support import ALEX, SARAH, create
from ask_support import citing, scripted
from test_ask_agenda import LEFT, agenda_of

from brain.api.deps import get_llm
from contracts import Answer, InvokeResponse, Source


@pytest.fixture
def llm(app):
    llm = scripted(answer=citing("Agenda:", text="Billing bug and Hiring plan are left."))
    app.dependency_overrides[get_llm] = lambda: llm
    return llm


def meeting_with_agenda(client_as, store) -> dict:
    meeting = create(client_as(ALEX), "Sprint review")
    asyncio.run(store.save_agenda(agenda_of(meeting["id"], person_id=SARAH.id)))
    return meeting


def agenda_source(meeting_id: str) -> Source:
    return Source(kind="meeting", label="Agenda", meeting_id=meeting_id)


def test_the_meeting_ask_cites_the_agenda(client_as, store, llm):
    meeting = meeting_with_agenda(client_as, store)

    response = client_as(SARAH).post(
        f"/meetings/{meeting['id']}/ask", json={"question": LEFT, "visibility": "private"}
    )

    assert response.status_code == 200, response.text
    answer = Answer.model_validate(response.json())
    assert answer.text == "Billing bug and Hiring plan are left."
    assert answer.sources == [agenda_source(meeting["id"])]


def test_the_workers_invoke_cites_the_agenda(client_as, worker, store, llm):
    meeting = meeting_with_agenda(client_as, store)
    invocation = {
        "id": "inv-1",
        "meeting_id": meeting["id"],
        "via": "voice",
        "visibility": "public",
        "asked_by_id": SARAH.id,
        "asked_by_name": SARAH.name,
        "question": LEFT,
    }

    response = worker.post(
        f"/internal/meetings/{meeting['id']}/invoke",
        json={"invocation": invocation, "recent_segments": []},
    )

    assert response.status_code == 200, response.text
    answer = InvokeResponse.model_validate(response.json()).answer
    assert answer.sources == [agenda_source(meeting["id"])]
    assert "Hiring plan (pending" in llm.calls[1].prompt


def test_home_ask_never_gets_an_agenda(app, client_as, store):
    meeting_with_agenda(client_as, store)
    llm = scripted()
    app.dependency_overrides[get_llm] = lambda: llm

    response = client_as(SARAH).post("/ask", json={"question": LEFT, "visibility": "private"})

    assert response.status_code == 200, response.text
    assert llm.calls and all("Hiring plan" not in call.prompt for call in llm.calls)
