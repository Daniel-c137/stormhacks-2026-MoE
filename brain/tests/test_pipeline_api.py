"""The write-up after the host ends a meeting: the saved final transcript in, a report to review
out. Requests go through the ASGI app on the test's own event loop, so a test can wait for the
background write-up with the runner's drain()."""

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM, WORKER_TOKEN, speakers_join
from pipeline_support import GatedLLM

from brain.agent.pipeline import NO_TRANSCRIPT, REPORT_STEPS, ReportPipeline
from brain.api.deps import (
    current_user,
    get_llm_factory,
    get_memory,
    get_pipeline,
    get_settings,
)
from brain.llm import LLMError, MockEmbedder, MockLLM
from brain.memory import InMemoryMemoryStore, MeetingMemory, UnusableMemory
from brain.report import ExtractedDecision, ExtractedTask, ReportExtraction
from brain.report.decisions import DecisionVerdict, DecisionVerdicts
from contracts import (
    AGENT_PARTICIPANT_ID,
    Decision,
    DecisionRelation,
    FactCheck,
    Person,
    Report,
    ReportProgress,
    Source,
    get_identity,
)

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
        said = segments(meeting_id, lines, first)
        await speakers_join(self.app, meeting_id, said)
        response = await self.http.post(
            f"/internal/meetings/{meeting_id}/segments",
            json={"segments": said},
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
def settings(settings):
    """No wait for the transcript to settle; the settle test gates it instead."""
    return settings.model_copy(update={"pipeline_settle_seconds": 0})


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
    assert done.pop("updated_at")
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


async def test_the_report_lists_polaris_only_when_it_joined_the_meeting(api, llm):
    agent_line = f"- {AGENT_PARTICIPANT_ID}: {get_identity().agent_name}"
    await ended(api, "Polaris never came")
    joined = await api.create("Polaris joined")
    response = await api.http.post(
        f"/internal/meetings/{joined['id']}/agent-joined",
        headers={"X-Internal-Token": WORKER_TOKEN},
    )
    assert response.status_code == 204
    await api.ingest(joined["id"])
    await api.end(joined["id"])
    await api.drain()

    absent, present = calls_for(llm, ReportExtraction)
    assert agent_line not in absent.prompt
    assert agent_line in present.prompt


async def test_final_segments_arriving_while_the_transcript_settles_are_written_up(
    api, app, store, llm, memory
):
    settled = asyncio.Event()
    app.dependency_overrides[get_pipeline] = lambda: ReportPipeline(
        store, lambda: llm, memory, settle=settled.wait
    )
    meeting = await api.create()
    await api.ingest(meeting["id"], LINES[:2])

    await api.end(meeting["id"])
    await api.ingest(meeting["id"], LINES[2:], first=3)  # the worker's last words, in flight
    settled.set()
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


async def test_the_meetings_confident_contradictions_are_given_to_the_report(api, store, llm):
    meeting = await api.create()
    await api.ingest(meeting["id"])
    claim = "The double charge fix from PR50 is already in the latest release."
    records = Source(kind="github_pr", label="dropsubs/app#50 merged after v0.9.3")
    checks = [
        FactCheck(
            id="fc-1",
            claim=claim,
            speaker_name=SARAH.name,
            verdict="contradicted",
            confidence=0.9,
            severity="high",
            sources=[records],
        ),
        FactCheck(
            id="fc-2",
            claim="The waitlist email went out on Monday.",
            speaker_name=ALEX.name,
            verdict="supported",
            confidence=0.9,
            severity="low",
        ),
    ]
    for check in checks:
        await store.add_fact_check(meeting["id"], check)

    await api.end(meeting["id"])
    await api.drain()

    (call,) = calls_for(llm, ReportExtraction)
    assert claim in call.prompt and records.label in call.prompt
    assert checks[1].claim not in call.prompt
    report = (await api.report(meeting["id"])).json()
    assert [c["id"] for c in report["fact_checks"]] == ["fc-1", "fc-2"]


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


async def test_an_older_meetings_write_up_never_retires_a_newer_meetings_decision(
    api, store, memory
):
    newer = await ended(api, "Refund follow-up")  # started now, written up first
    held_id = f"{newer['id']}-decision-1"
    older = await api.create("Refund sync")
    started = datetime.fromisoformat(newer["started_at"]) - timedelta(days=2)
    stored = await store.meeting(older["id"])
    await store.update_meeting(stored.model_copy(update={"started_at": started}))
    llm = MockLLM(
        structured={
            ReportExtraction: EXTRACTION,
            DecisionVerdicts: DecisionVerdicts(
                verdicts=[
                    DecisionVerdict(
                        decision_id=f"{older['id']}-decision-1",
                        verdict="contradicts",
                        past_decision_id=held_id,
                        reason="Backwards in time.",
                        confidence=0.95,
                    )
                ]
            ),
        }
    )
    api.use_llm(llm)
    assert any(  # memory would offer the newer decision if it were allowed
        h.chunk.ref_id == held_id for h in await memory.search(TEAM.id, "waitlist email", k=50)
    )

    await api.ingest(older["id"])
    await api.end(older["id"])
    await api.drain()

    assert await api.status(older["id"]) == "needs_review"
    decisions = {d["id"]: d for d in (await api.get("/decisions")).json()}
    assert (decisions[held_id]["status"], decisions[held_id]["relation"]) == ("active", None)
    assert decisions[f"{older['id']}-decision-1"]["relation"] is None
    assert calls_for(llm, DecisionVerdicts) == []


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
    (window,) = [c for c in mine if c.kind == "transcript"]
    assert f"{SARAH.name}: The fix is merged." in window.text.splitlines()[1]
    assert window.speaker_id is None  # two people speak in it


async def test_the_agents_own_words_are_left_out_of_memory(api, memory):
    agent = get_identity().agent_name
    said_by_agent = "Jira still shows the refund ticket as In Progress."
    agent_person = Person(id=AGENT_PARTICIPANT_ID, name=agent, short=agent, initials=agent[:1])
    meeting = await api.create()
    await api.ingest(meeting["id"])
    await api.ingest(meeting["id"], [(agent_person, said_by_agent)], first=len(LINES) + 1)
    await api.end(meeting["id"])
    await api.drain()

    hits = await memory.search(TEAM.id, said_by_agent, k=50)

    mine = [h.chunk for h in hits if h.chunk.meeting_id == meeting["id"]]
    assert any(c.kind == "transcript" for c in mine)
    assert all(c.speaker_id != AGENT_PARTICIPANT_ID for c in mine)
    assert not any(said_by_agent in c.text for c in mine)


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
    assert "language model failed" in progress["error"]
    assert "ReportExtraction" not in progress["error"]  # the raw exception stays in the log
    assert (await api.report(meeting["id"])).status_code == 404
    assert (await api.get("/tasks")).json() == []
    assert (await api.get("/decisions")).json() == []
    assert await memory.search(TEAM.id, "refund", k=20) == []


async def test_a_meeting_without_a_transcript_goes_to_review_with_an_empty_report(api, llm, memory):
    meeting = await api.create()

    await api.end(meeting["id"])
    await api.drain()

    assert await api.status(meeting["id"]) == "needs_review"
    report = (await api.report(meeting["id"])).json()
    assert report["summary"] == NO_TRANSCRIPT
    assert report["tasks"] == report["decisions"] == report["topics"] == []
    progress = await api.progress(meeting["id"])
    assert (progress["done"], progress["error"]) == (True, None)
    assert llm.calls == []
    assert await memory.search(TEAM.id, "anything", k=20) == []


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
    body = response.json()
    assert body.pop("updated_at")
    assert body == {
        "meeting_id": meeting["id"],
        "steps": REPORT_STEPS,
        "current": 0,
        "done": False,
        "error": None,
    }
    await api.drain()
    assert await api.status(meeting["id"]) == "needs_review"
    assert (await api.report(meeting["id"])).json()["summary"] == EXTRACTION.summary


async def test_retry_is_refused_unless_the_last_run_failed_or_stalled(api):
    gated = GatedLLM(structured={ReportExtraction: EXTRACTION})
    api.use_llm(gated)
    meeting = await api.create()
    await api.ingest(meeting["id"])

    live = await api.retry(meeting["id"])
    assert (live.status_code, live.json()["detail"]) == (
        409,
        "The meeting is live, not being written up",
    )
    await api.end(meeting["id"])
    await gated.called.wait()
    before = await api.progress(meeting["id"])
    running = await api.retry(meeting["id"])
    assert (running.status_code, running.json()["detail"]) == (
        409,
        "The write-up is still running",
    )
    assert await api.progress(meeting["id"]) == before  # a refused retry resets nothing
    gated.gate.set()
    await api.drain()
    done = await api.retry(meeting["id"])
    assert (done.status_code, done.json()["detail"]) == (
        409,
        "The meeting is needs_review, not being written up",
    )

    assert len(calls_for(gated, ReportExtraction)) == 1


async def stalled(api: Api, store, *, minutes_ago: float | None):
    """A processing meeting whose write-up nobody in this process is running, as after a hard
    crash: its last progress is `minutes_ago` old (None: no progress was ever saved)."""
    meeting = await api.create()
    await api.ingest(meeting["id"])
    await store.transition_status(meeting["id"], {"live"}, "processing")
    if minutes_ago is not None:
        await store.save_report_progress(
            ReportProgress(
                meeting_id=meeting["id"],
                steps=REPORT_STEPS,
                current=1,
                done=False,
                updated_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
            )
        )
    return meeting


async def test_a_write_up_that_stopped_moving_can_be_retried(api, store):
    meeting = await stalled(api, store, minutes_ago=11)

    response = await api.retry(meeting["id"])
    await api.drain()

    assert response.status_code == 202
    assert await api.status(meeting["id"]) == "needs_review"


async def test_a_write_up_that_never_saved_progress_can_be_retried(api, store):
    meeting = await stalled(api, store, minutes_ago=None)

    assert (await api.retry(meeting["id"])).status_code == 202
    await api.drain()
    assert await api.status(meeting["id"]) == "needs_review"


async def test_a_write_up_still_moving_elsewhere_is_not_retried(api, store, app, settings):
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"pipeline_stale_minutes": 30}
    )
    meeting = await stalled(api, store, minutes_ago=11)
    before = await api.progress(meeting["id"])

    response = await api.retry(meeting["id"])

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "The write-up is still in progress; it can be retried once it fails or makes no"
        " progress for 30 minutes"
    )
    assert await api.progress(meeting["id"]) == before


# what a failure leaves behind


async def test_if_the_write_up_cannot_be_scheduled_the_end_still_succeeds_and_can_be_retried(
    api, app, monkeypatch
):
    runner = app.state.pipeline_runner
    start = runner.start

    def broken(meeting_id, run):
        monkeypatch.setattr(runner, "start", start)
        raise RuntimeError("no event loop")

    monkeypatch.setattr(runner, "start", broken)
    meeting = await api.create()
    await api.ingest(meeting["id"])

    response = await api.end(meeting["id"])

    assert (response.status_code, response.json()["status"]) == (200, "processing")
    assert "could not be started" in (await api.progress(meeting["id"]))["error"]
    assert (await api.retry(meeting["id"])).status_code == 202
    await api.drain()
    assert await api.status(meeting["id"]) == "needs_review"


async def test_a_failed_first_progress_write_is_recorded_and_can_be_retried(
    api, store, monkeypatch
):
    save = store.save_report_progress
    calls = []

    async def first_fails(progress):
        calls.append(progress)
        if len(calls) == 1:
            raise RuntimeError("connection to server at 10.0.0.5, port 5432 failed")
        return await save(progress)

    monkeypatch.setattr(store, "save_report_progress", first_fails)
    meeting = await api.create()
    await api.ingest(meeting["id"])

    response = await api.end(meeting["id"])
    await api.drain()

    assert response.status_code == 200
    error = (await api.progress(meeting["id"]))["error"]
    assert error and "10.0.0.5" not in error
    assert (await api.retry(meeting["id"])).status_code == 202
    await api.drain()
    assert await api.status(meeting["id"]) == "needs_review"


async def test_a_failed_link_step_leaves_only_the_error(api, store, memory):
    send = past_decision("d-send", "Send the waitlist email as soon as the fix merges")
    await past_meeting(store, "Launch planning", send)
    api.use_llm(MockLLM(structured={ReportExtraction: EXTRACTION}))  # no verdicts scripted

    meeting = await ended(api)

    progress = await api.progress(meeting["id"])
    linking = REPORT_STEPS.index("Checking against past decisions")
    assert (progress["current"], progress["done"]) == (linking, False)
    assert "language model failed" in progress["error"]
    assert await api.status(meeting["id"]) == "processing"
    assert (await api.report(meeting["id"])).status_code == 404
    decisions = (await api.get("/decisions")).json()
    assert [(d["id"], d["status"], d["relation"]) for d in decisions] == [
        ("d-send", "active", None)
    ]
    assert await memory.search(TEAM.id, "refund", k=20) == []


def contradicting(meeting_id: str) -> MockLLM:
    return MockLLM(
        structured={
            ReportExtraction: EXTRACTION,
            DecisionVerdicts: DecisionVerdicts(
                verdicts=[
                    DecisionVerdict(
                        decision_id=f"{meeting_id}-decision-1",
                        verdict="contradicts",
                        past_decision_id="d-send",
                        reason="Holding the email reverses sending it at once.",
                        confidence=0.9,
                    )
                ]
            ),
        }
    )


async def test_a_failed_save_leaves_no_report_links_or_memory_and_a_retry_links_once(
    api, store, memory, monkeypatch
):
    send = past_decision("d-send", "Send the waitlist email as soon as the fix merges")
    await past_meeting(store, "Launch planning", send)
    meeting = await api.create()
    new_id = f"{meeting['id']}-decision-1"
    api.use_llm(contradicting(meeting["id"]))
    complete = store.complete_report

    async def fails_once(*args, **kwargs):
        monkeypatch.setattr(store, "complete_report", complete)
        raise RuntimeError("connection to server at 10.0.0.5, port 5432 failed")

    monkeypatch.setattr(store, "complete_report", fails_once)
    await api.ingest(meeting["id"])
    await api.end(meeting["id"])
    await api.drain()

    progress = await api.progress(meeting["id"])
    assert progress["current"] == REPORT_STEPS.index("Saving the report")
    assert "Saving the report" in progress["error"] and "10.0.0.5" not in progress["error"]
    assert await api.status(meeting["id"]) == "processing"
    assert (await api.report(meeting["id"])).status_code == 404
    assert [d["status"] for d in (await api.get("/decisions")).json()] == ["active"]
    assert await memory.search(TEAM.id, "refund", k=20) == []

    assert (await api.retry(meeting["id"])).status_code == 202
    await api.drain()

    assert await api.status(meeting["id"]) == "needs_review"
    decisions = {d["id"]: d for d in (await api.get("/decisions")).json()}
    assert decisions["d-send"]["status"] == "superseded"
    assert decisions["d-send"]["relation"] == {"type": "superseded_by", "decision_id": new_id}
    assert decisions[new_id]["relation"] == {"type": "contradicts", "decision_id": "d-send"}
    assert len((await api.get("/tasks")).json()) == 1
    # the three turns' window, the summary, one decision and one task, each once
    mine = [h.chunk for h in await memory.search(TEAM.id, "refund", k=100)]
    assert sorted(c.kind for c in mine if c.meeting_id == meeting["id"]) == sorted(
        ["transcript", "summary", "decision", "task"]
    )


async def half_saved_by_an_earlier_run(api, store) -> tuple[dict, str]:
    """What the first version of the write-up could leave: the report saved, d-send retired
    by this meeting's decision, and the meeting still processing with an error."""
    send = past_decision("d-send", "Send the waitlist email as soon as the fix merges")
    await past_meeting(store, "Launch planning", send)
    meeting = await api.create()
    await api.ingest(meeting["id"])
    await store.transition_status(meeting["id"], {"live"}, "processing")
    new_id = f"{meeting['id']}-decision-1"
    held = Decision(
        id=new_id,
        meeting_id=meeting["id"],
        text="Hold the waitlist email",
        made_by=ALEX.name,
        t=12,
        quote=LINES[2][1],
        relation=DecisionRelation(type="contradicts", decision_id="d-send"),
    )
    await store.save_report(Report(meeting_id=meeting["id"], summary="x", decisions=[held]))
    (old,) = [d for d in await store.decisions(TEAM.id) if d.id == "d-send"]
    retired = DecisionRelation(type="superseded_by", decision_id=new_id)
    await store.update_decision(
        old.model_copy(update={"status": "superseded", "relation": retired})
    )
    await store.save_report_progress(
        ReportProgress(
            meeting_id=meeting["id"], steps=REPORT_STEPS, current=3, done=False, error="failed"
        )
    )
    return meeting, new_id


async def test_a_retry_relinks_a_past_decision_an_earlier_partial_run_retired(api, store):
    meeting, new_id = await half_saved_by_an_earlier_run(api, store)
    api.use_llm(contradicting(meeting["id"]))

    assert (await api.retry(meeting["id"])).status_code == 202
    await api.drain()

    decisions = {d["id"]: d for d in (await api.get("/decisions")).json()}
    assert decisions["d-send"]["relation"] == {"type": "superseded_by", "decision_id": new_id}
    assert decisions[new_id]["relation"] == {"type": "contradicts", "decision_id": "d-send"}


async def test_a_retry_that_finds_no_decision_restores_the_retired_past_decision(api, store):
    meeting, _ = await half_saved_by_an_earlier_run(api, store)
    no_decisions = EXTRACTION.model_copy(update={"decisions": []})
    api.use_llm(MockLLM(structured={ReportExtraction: no_decisions}))

    assert (await api.retry(meeting["id"])).status_code == 202
    await api.drain()

    decisions = (await api.get("/decisions")).json()
    assert [(d["id"], d["status"], d["relation"]) for d in decisions] == [
        ("d-send", "active", None)
    ]


class FailingIndex(MeetingMemory):
    async def index_meeting(self, *args, **kwargs):
        raise LLMError("embedding service at 10.0.0.9 returned 500")


async def test_a_failed_index_still_reaches_review_and_says_indexing_failed(api):
    api.use_memory(FailingIndex(MockEmbedder(), InMemoryMemoryStore(dim=768)))

    meeting = await ended(api)

    assert await api.status(meeting["id"]) == "needs_review"
    assert (await api.report(meeting["id"])).status_code == 200
    progress = await api.progress(meeting["id"])
    assert (progress["done"], progress["error"]) == (True, None)
    (indexing,) = [s for s in progress["steps"] if s.startswith("Indexing for search")]
    assert "failed" in indexing and "10.0.0.9" not in indexing


async def test_unusable_memory_leaves_linking_to_lexical_candidates(api, store):
    send = past_decision("d-send", "Send the waitlist email as soon as the fix merges")
    await past_meeting(store, "Launch planning", send)
    llm = MockLLM(structured={ReportExtraction: EXTRACTION, DecisionVerdicts: DecisionVerdicts()})
    api.use_llm(llm)
    api.use_memory(UnusableMemory("the embedder makes 1536-dimension vectors; the store holds 768"))

    meeting = await ended(api)

    assert await api.status(meeting["id"]) == "needs_review"
    (call,) = calls_for(llm, DecisionVerdicts)
    assert "d-send" in call.prompt  # shares "waitlist email" with the new decision


async def test_misconfigured_memory_says_so_in_the_index_step(api):
    api.use_memory(UnusableMemory("the embedder makes 1536-dimension vectors; the store holds 768"))

    meeting = await ended(api)

    assert await api.status(meeting["id"]) == "needs_review"
    progress = await api.progress(meeting["id"])
    (indexing,) = [s for s in progress["steps"] if s.startswith("Indexing for search")]
    assert "misconfigured" in indexing and "1536" in indexing
    assert "not configured" not in indexing
