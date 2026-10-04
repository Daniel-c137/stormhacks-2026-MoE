"""The write-up after the host ends a meeting: the saved final transcript in, a report to review
out. Requests go through the ASGI app on the test's own event loop, so a test can wait for the
background write-up with the runner's drain()."""

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM, WORKER_TOKEN
from pipeline_support import GatedLLM

from brain.agent.pipeline import REPORT_STEPS
from brain.api.deps import current_user, get_llm_factory, get_memory, get_settings
from brain.llm import MockEmbedder, MockLLM
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.report import ExtractedDecision, ExtractedTask, ReportExtraction
from brain.report.decisions import DecisionVerdict, DecisionVerdicts
from contracts import Decision, Person, Report

pytestmark = pytest.mark.anyio

LINES: list[tuple[Person, str]] = [
    (ALEX, "Morning, quick sync on the double charge."),
    (SARAH, "The fix is merged. I'll refund the 14 affected users by Wednesday."),
    (ALEX, "Then we hold the waitlist email until v0.9.4 ships. Agreed."),
]

EXTRACTION = ReportExtraction(
    summary="Sarah will refund the affected users; the waitlist email waits for v0.9.4.",
    topics=["Double charge", "Waitlist email"],
    decisions=[
        ExtractedDecision(
            text="Hold the waitlist email until v0.9.4 ships",
            made_by_id=ALEX.id,
            evidence=["s3"],
        )
    ],
    tasks=[ExtractedTask(title="Refund the 14 affected users", owner_id=SARAH.id, evidence=["s2"])],
)


def segments(meeting_id: str, lines=LINES, first: int = 1) -> list[dict]:
    """Final segments six seconds apart, numbered from `first`."""
    return [
        {
            "seg_id": f"{meeting_id}-seg-{n}",
            "meeting_id": meeting_id,
            "speaker_id": person.id,
            "speaker_name": person.name,
            "text": text,
            "is_final": True,
            "t_start": (n - 1) * 6.0,
            "t_end": (n - 1) * 6.0 + 5,
        }
        for n, (person, text) in enumerate(lines, first)
    ]


class Api:
    """The board, the worker and the background runner of one app."""

    def __init__(self, app, http: httpx.AsyncClient):
        self.app = app
        self.http = http

    def use_llm(self, llm) -> None:
        self.app.dependency_overrides[get_llm_factory] = lambda: lambda: llm

    def use_memory(self, memory: MeetingMemory | None) -> None:
        self.app.dependency_overrides[get_memory] = lambda: memory

    async def call(self, method: str, path: str, person: Person = ALEX, **kwargs):
        self.app.dependency_overrides[current_user] = lambda: person
        return await self.http.request(method, path, **kwargs)

    async def get(self, path: str, person: Person = ALEX) -> httpx.Response:
        return await self.call("GET", path, person)

    async def create(self, title: str = "Refund sync") -> dict:
        response = await self.call("POST", "/meetings", json={"title": title})
        assert response.status_code == 200, response.text
        return response.json()

    async def ingest(self, meeting_id: str, lines=LINES, first: int = 1) -> None:
        response = await self.http.post(
            f"/internal/meetings/{meeting_id}/segments",
            json={"segments": segments(meeting_id, lines, first)},
            headers={"X-Internal-Token": WORKER_TOKEN},
        )
        assert response.status_code == 204, response.text

    async def end(self, meeting_id: str, person: Person = ALEX) -> httpx.Response:
        return await self.call("POST", f"/meetings/{meeting_id}/end", person)

    async def retry(self, meeting_id: str, person: Person = ALEX) -> httpx.Response:
        return await self.call("POST", f"/meetings/{meeting_id}/report/retry", person)

    async def status(self, meeting_id: str) -> str:
        return (await self.get(f"/meetings/{meeting_id}")).json()["status"]

    async def progress(self, meeting_id: str) -> dict:
        response = await self.get(f"/meetings/{meeting_id}/report/progress")
        assert response.status_code == 200, response.text
        return response.json()

    async def report(self, meeting_id: str) -> httpx.Response:
        return await self.get(f"/meetings/{meeting_id}/report")

    async def drain(self) -> None:
        await self.app.state.pipeline_runner.drain()


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM(structured={ReportExtraction: EXTRACTION})


@pytest.fixture
def memory() -> MeetingMemory:
    return MeetingMemory(MockEmbedder(), InMemoryMemoryStore(dim=768))


@pytest.fixture
async def api(app, llm, memory):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://brain") as http:
        api = Api(app, http)
        api.use_llm(llm)
        api.use_memory(memory)
        yield api


async def ended(api: Api, title: str = "Refund sync") -> dict:
    """A meeting with the standard lines, ended and written up."""
    meeting = await api.create(title)
    await api.ingest(meeting["id"])
    assert (await api.end(meeting["id"])).status_code == 200
    await api.drain()
    return meeting


def calls_for(llm: MockLLM, schema) -> list:
    return [c for c in llm.calls if c.schema is schema]


async def past_meeting(store, title: str, *decisions: Decision, day=datetime(2026, 9, 18, 16)):
    """An earlier meeting of the team, already written up with these decisions."""
    meeting = await store.create_meeting(TEAM.id, title, ALEX.id)
    meeting = await store.update_meeting(
        meeting.model_copy(update={"status": "needs_review", "started_at": day.replace(tzinfo=UTC)})
    )
    saved = [d.model_copy(update={"meeting_id": meeting.id}) for d in decisions]
    await store.save_report(Report(meeting_id=meeting.id, summary="Earlier.", decisions=saved))
    return meeting, saved


def past_decision(id: str, text: str) -> Decision:
    return Decision(id=id, meeting_id="?", text=text, made_by=SARAH.name, t=30, quote=text)


# the run


async def test_ending_returns_at_once_and_the_meeting_is_written_up_for_review(api):
    gated = GatedLLM(structured={ReportExtraction: EXTRACTION})
    api.use_llm(gated)
    meeting = await api.create()
    await api.ingest(meeting["id"])

    response = await api.end(meeting["id"])

    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    await gated.called.wait()  # the end request returned while the report is still being written
    running = await api.progress(meeting["id"])
    assert (running["current"], running["done"], running["error"]) == (1, False, None)
    assert (await api.report(meeting["id"])).status_code == 404

    gated.gate.set()
    await api.drain()

    assert await api.status(meeting["id"]) == "needs_review"
    report = (await api.report(meeting["id"])).json()
    assert report["summary"] == EXTRACTION.summary
    (task,) = report["tasks"]
    assert (task["title"], task["owner_id"], task["quote"]) == (
        "Refund the 14 affected users",
        SARAH.id,
        LINES[1][1],
    )
    (decision,) = report["decisions"]
    assert (decision["made_by"], decision["quote"]) == (ALEX.name, LINES[2][1])
    done = await api.progress(meeting["id"])
    assert done == {
        "meeting_id": meeting["id"],
        "steps": REPORT_STEPS,
        "current": len(REPORT_STEPS),
        "done": True,
        "error": None,
    }


async def test_progress_is_saved_after_each_step(api, store, monkeypatch):
    saved = []
    save = store.save_report_progress

    async def spy(progress):
        saved.append(progress)
        return await save(progress)

    monkeypatch.setattr(store, "save_report_progress", spy)

    await ended(api)

    assert all(p.steps == REPORT_STEPS and p.error is None for p in saved)
    assert list(dict.fromkeys(p.current for p in saved)) == list(range(len(REPORT_STEPS) + 1))
    assert [p.done for p in saved] == [False] * (len(saved) - 1) + [True]


async def test_the_report_is_written_from_the_meetings_title_date_members_and_transcript(api, llm):
    meeting = await ended(api, "Refund sync")

    (call,) = calls_for(llm, ReportExtraction)
    started = datetime.fromisoformat(meeting["started_at"]).date().isoformat()
    for expected in (
        "Meeting: Refund sync",
        f"Date: {started}",
        f"- {ALEX.id}: {ALEX.name}",
        f"- {SARAH.id}: {SARAH.name}",
        f"[s2 00:06] {SARAH.name}: {LINES[1][1]}",
    ):
        assert expected in call.prompt


async def test_final_segments_arriving_while_the_transcript_settles_are_written_up(
    api, app, settings, llm
):
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"pipeline_settle_seconds": 0.3}
    )
    meeting = await api.create()
    await api.ingest(meeting["id"], LINES[:2])

    await api.end(meeting["id"])
    await api.ingest(meeting["id"], LINES[2:], first=3)  # the worker's last words, in flight
    await api.drain()

    (call,) = calls_for(llm, ReportExtraction)
    assert LINES[2][1] in call.prompt
    report = (await api.report(meeting["id"])).json()
    assert [d["quote"] for d in report["decisions"]] == [LINES[2][1]]


async def test_the_agenda_is_given_to_the_report_in_order(api, llm):
    meeting = await api.create()
    response = await api.call(
        "PUT",
        f"/meetings/{meeting['id']}/agenda",
        json={
            "items": [
                {"title": "Waitlist email", "minutes": 5},
                {"title": "Double charge refunds", "minutes": 10},
            ]
        },
    )
    assert response.status_code == 200, response.text
    await api.ingest(meeting["id"])

    await api.end(meeting["id"])
    await api.drain()

    (call,) = calls_for(llm, ReportExtraction)
    first = call.prompt.index("1. Waitlist email (5 min)")
    assert first < call.prompt.index("2. Double charge refunds (10 min)")


async def test_a_meeting_without_an_agenda_has_none_in_the_prompt(api, llm):
    await ended(api)

    (call,) = calls_for(llm, ReportExtraction)
    assert "Agenda" not in call.prompt


# past decisions


async def test_new_decisions_are_linked_to_past_decisions_of_other_meetings(api, store):
    send = past_decision("d-send", "Send the waitlist email as soon as the fix merges")
    await past_meeting(store, "Launch planning", send)
    meeting = await api.create()
    new_id = f"{meeting['id']}-decision-1"
    llm = MockLLM(
        structured={
            ReportExtraction: EXTRACTION,
            DecisionVerdicts: DecisionVerdicts(
                verdicts=[
                    DecisionVerdict(
                        decision_id=new_id,
                        verdict="contradicts",
                        past_decision_id="d-send",
                        reason="Holding the email reverses sending it at once.",
                        confidence=0.9,
                    )
                ]
            ),
        }
    )
    api.use_llm(llm)
    await api.ingest(meeting["id"])

    await api.end(meeting["id"])
    await api.drain()

    decisions = {d["id"]: d for d in (await api.get("/decisions")).json()}
    assert decisions["d-send"]["status"] == "superseded"
    assert decisions["d-send"]["relation"] == {"type": "superseded_by", "decision_id": new_id}
    assert decisions[new_id]["relation"] == {"type": "contradicts", "decision_id": "d-send"}
    (call,) = calls_for(llm, DecisionVerdicts)
    assert "2026-09-18" in call.prompt  # the past decision is dated by its meeting


async def test_past_decisions_found_in_memory_are_offered_even_without_shared_words(
    api, store, memory
):
    beta = past_decision("d-beta", "Announce the beta to everyone on Monday")
    earlier, (beta,) = await past_meeting(store, "Launch planning", beta)
    await memory.index_meeting(
        TEAM.id, earlier.id, [], Report(meeting_id=earlier.id, summary="", decisions=[beta])
    )
    llm = MockLLM(structured={ReportExtraction: EXTRACTION, DecisionVerdicts: DecisionVerdicts()})
    api.use_llm(llm)

    await ended(api)

    (call,) = calls_for(llm, DecisionVerdicts)
    assert "d-beta" in call.prompt


async def test_without_memory_only_past_decisions_sharing_words_are_offered(api, store):
    await past_meeting(
        store, "Launch planning", past_decision("d-beta", "Announce the beta to everyone")
    )
    llm = MockLLM(structured={ReportExtraction: EXTRACTION, DecisionVerdicts: DecisionVerdicts()})
    api.use_llm(llm)
    api.use_memory(None)

    await ended(api)

    assert calls_for(llm, DecisionVerdicts) == []


# memory


async def test_the_transcript_and_report_are_indexed_into_memory(api, memory):
    meeting = await ended(api)

    hits = await memory.search(TEAM.id, "refund the affected users", k=20)

    mine = [h.chunk for h in hits if h.chunk.meeting_id == meeting["id"]]
    assert {c.kind for c in mine} == {"transcript", "summary", "decision", "task"}
    turns = [c for c in mine if c.kind == "transcript"]
    assert turns[0].speaker_id == SARAH.id


async def test_without_embeddings_the_write_up_finishes_and_says_indexing_was_skipped(api):
    api.use_memory(None)

    meeting = await ended(api)

    assert await api.status(meeting["id"]) == "needs_review"
    progress = await api.progress(meeting["id"])
    assert progress["done"] is True
    (indexing,) = [s for s in progress["steps"] if s.startswith("Indexing for search")]
    assert "skipped" in indexing and "not configured" in indexing


# once only, failures and retries


async def test_ending_twice_at_once_writes_up_once(api):
    gated = GatedLLM(structured={ReportExtraction: EXTRACTION})
    api.use_llm(gated)  # held, so both ends answer while the write-up is still running
    meeting = await api.create()
    await api.ingest(meeting["id"])

    first, second = await asyncio.gather(api.end(meeting["id"]), api.end(meeting["id"]))
    gated.gate.set()
    await api.drain()
    again = await api.end(meeting["id"])
    await api.drain()

    assert [r.json()["status"] for r in (first, second)] == ["processing", "processing"]
    assert again.json()["status"] == "needs_review"
    assert len(calls_for(gated, ReportExtraction)) == 1


async def test_a_failed_step_keeps_the_meeting_processing_with_the_error_and_saves_nothing(
    api, memory
):
    api.use_llm(MockLLM())  # no scripted report: the model call fails

    meeting = await ended(api)

    assert await api.status(meeting["id"]) == "processing"
    progress = await api.progress(meeting["id"])
    assert (progress["current"], progress["done"]) == (1, False)
    assert "no scripted response for ReportExtraction" in progress["error"]
    assert (await api.report(meeting["id"])).status_code == 404
    assert (await api.get("/tasks")).json() == []
    assert (await api.get("/decisions")).json() == []
    assert await memory.search(TEAM.id, "refund", k=20) == []


async def test_a_meeting_without_a_transcript_fails_with_a_clear_error(api, llm):
    meeting = await api.create()

    await api.end(meeting["id"])
    await api.drain()

    assert await api.status(meeting["id"]) == "processing"
    assert "transcript" in (await api.progress(meeting["id"]))["error"]
    assert llm.calls == []


async def test_without_gemini_the_meeting_still_ends_and_the_write_up_says_why(api, app):
    del app.dependency_overrides[get_llm_factory]  # the real factory; the settings have no Gemini
    meeting = await api.create()
    await api.ingest(meeting["id"])

    response = await api.end(meeting["id"])
    await api.drain()

    assert response.status_code == 200
    assert "Gemini is not configured" in (await api.progress(meeting["id"]))["error"]


async def test_the_host_retries_a_failed_write_up(api, llm):
    api.use_llm(MockLLM())
    meeting = await ended(api)
    api.use_llm(llm)

    assert (await api.retry(meeting["id"], SARAH)).status_code == 403
    assert (await api.retry(meeting["id"], OUTSIDER)).status_code == 404
    response = await api.retry(meeting["id"])

    assert response.status_code == 202
    assert response.json() == {
        "meeting_id": meeting["id"],
        "steps": REPORT_STEPS,
        "current": 0,
        "done": False,
        "error": None,
    }
    await api.drain()
    assert await api.status(meeting["id"]) == "needs_review"
    assert (await api.report(meeting["id"])).json()["summary"] == EXTRACTION.summary


async def test_retry_is_refused_unless_the_last_run_failed(api):
    gated = GatedLLM(structured={ReportExtraction: EXTRACTION})
    api.use_llm(gated)
    meeting = await api.create()
    await api.ingest(meeting["id"])

    assert (await api.retry(meeting["id"])).status_code == 409  # still live
    await api.end(meeting["id"])
    await gated.called.wait()
    assert (await api.retry(meeting["id"])).status_code == 409  # still running
    gated.gate.set()
    await api.drain()
    assert (await api.retry(meeting["id"])).status_code == 409  # written up

    assert len(calls_for(gated, ReportExtraction)) == 1


async def test_a_retry_replaces_the_report_and_memory_instead_of_adding_to_them(
    api, store, memory, monkeypatch
):
    save = store.save_report
    attempts = []

    async def flaky(report):
        attempts.append(report)
        if len(attempts) == 1:
            raise RuntimeError("the database went away")
        await save(report)

    monkeypatch.setattr(store, "save_report", flaky)
    meeting = await ended(api)
    failed = await api.progress(meeting["id"])
    assert failed["error"] == "the database went away"
    assert failed["current"] == len(REPORT_STEPS) - 1
    indexed = await memory.search(TEAM.id, "refund", k=100)

    assert (await api.retry(meeting["id"])).status_code == 202
    await api.drain()

    assert await api.status(meeting["id"]) == "needs_review"
    assert len(await memory.search(TEAM.id, "refund", k=100)) == len(indexed)
    assert len((await api.get("/tasks")).json()) == 1
    assert len((await api.get("/decisions")).json()) == 1
