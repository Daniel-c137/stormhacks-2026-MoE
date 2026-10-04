"""Live: the standup ingested through the API and ended by its host; real Gemini writes it up
(with the fallback chain) and real Gemini embeddings index it into an in-memory vector store.
The meeting has an agenda with an item nobody discusses, and the team has an earlier decision
the standup reverses, so the agenda block and past-decision linking both meet a real model.
Needs GEMINI_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM.
Deselected unless pytest runs with `-m live`."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from brain.api.deps import current_user, get_llm_factory, get_memory, get_settings, get_store
from brain.config import Settings
from brain.llm import GeminiEmbedder, GeminiLLM, make_embedder, make_llm
from brain.main import create_app
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.report import TranscriptInput
from brain.store import InMemoryStore
from contracts import Decision, Report, Team

FIXTURES = Path(__file__).parent.parent / "fixtures"
WORKER_TOKEN = "live-worker-token"
settings = Settings(openrouter_models=None)  # Gemini alone, never the fallback

pytestmark = [
    pytest.mark.live,
    pytest.mark.anyio,
    pytest.mark.skipif(
        not (
            settings.gemini_api_key
            and settings.gemini_model
            and settings.gemini_embedding_model
            and settings.gemini_embedding_dim
        ),
        reason="set GEMINI_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM",
    ),
]


async def test_gemini_writes_up_the_ended_standup_and_indexes_it():
    standup = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    team = Team(id="t-live", name="DropSubs", member_ids=[p.id for p in standup.members])
    host = standup.members[0]
    store = InMemoryStore(teams=[team], people=standup.members)
    llm = make_llm(settings)
    embedder = make_embedder(settings)
    assert isinstance(llm, GeminiLLM) and isinstance(embedder, GeminiEmbedder)
    memory = MeetingMemory(embedder, InMemoryMemoryStore(dim=embedder.dim))
    earlier = await store.create_meeting(team.id, "Launch planning", host.id)
    earlier = await store.update_meeting(
        earlier.model_copy(
            update={
                "status": "needs_review",
                "started_at": datetime(2026, 9, 25, 9, 30, tzinfo=UTC),
            }
        )
    )
    send_now = "Send the waitlist email as soon as the email exploit fix is merged."
    await store.save_report(
        Report(
            meeting_id=earlier.id,
            summary="Launch planning.",
            decisions=[
                Decision(
                    id="d-send-now",
                    meeting_id=earlier.id,
                    text="Send the waitlist email as soon as the exploit fix merges",
                    made_by="Alice Moreau",
                    t=120,
                    quote=send_now,
                )
            ],
        )
    )

    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"brain_internal_token": WORKER_TOKEN, "pipeline_settle_seconds": 0}
    )
    app.dependency_overrides[current_user] = lambda: host
    app.dependency_overrides[get_llm_factory] = lambda: lambda: llm
    app.dependency_overrides[get_memory] = lambda: memory

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://brain") as http:
        meeting = (await http.post("/meetings", json={"title": standup.title})).json()
        agenda = await http.put(
            f"/meetings/{meeting['id']}/agenda",
            json={
                "items": [
                    {"title": "Double-charge refunds", "minutes": 5},
                    {"title": "Waitlist email timing", "minutes": 5},
                    {"title": "Office move logistics", "minutes": 10},
                ]
            },
        )
        assert agenda.status_code == 200, agenda.text
        segments = [
            s.model_copy(
                update={"meeting_id": meeting["id"], "seg_id": f"{meeting['id']}-{s.seg_id}"}
            ).model_dump(mode="json")
            for s in standup.segments
        ]
        ingested = await http.post(
            f"/internal/meetings/{meeting['id']}/segments",
            json={"segments": segments},
            headers={"X-Internal-Token": WORKER_TOKEN},
        )
        assert ingested.status_code == 204, ingested.text

        ended = await http.post(f"/meetings/{meeting['id']}/end")
        assert ended.json()["status"] == "processing"
        await app.state.pipeline_runner.drain()

        progress = (await http.get(f"/meetings/{meeting['id']}/report/progress")).json()
        print(f"answered by {llm.last_model} (chain: {', '.join(llm.models)}); {progress}")
        assert progress["error"] is None and progress["done"] is True
        status = (await http.get(f"/meetings/{meeting['id']}")).json()["status"]
        assert status == "needs_review"
        report = Report.model_validate((await http.get(f"/meetings/{meeting['id']}/report")).json())

    # Quotes and owners are checked by the grounding whatever the model says; these are not.
    print(f"topics: {report.topics}")
    print(f"decisions: {[(d.text, d.relation) for d in report.decisions]}")
    assert report.summary
    refund = [t for t in report.tasks if "refund" in t.title.lower()]
    assert refund and refund[0].owner_id == "p-bob"
    assert report.topics, "the standup covered several agenda items"
    assert not any("office" in topic.lower() for topic in report.topics)
    (past,) = [d for d in await store.decisions(team.id) if d.id == "d-send-now"]
    print(f"past decision: {past.status} {past.relation}")
    assert past.status == "superseded"
    assert past.relation and past.relation.decision_id in {d.id for d in report.decisions}

    hits = await memory.search(team.id, "refund", k=10)
    for hit in hits:
        print(f"{hit.score:.3f} {hit.chunk.kind} {hit.chunk.text}")
    turns = [h.chunk for h in hits if h.chunk.kind == "transcript"]
    assert turns and turns[0].speaker_id == "p-bob"
    assert "refund the 14 affected users" in turns[0].text
