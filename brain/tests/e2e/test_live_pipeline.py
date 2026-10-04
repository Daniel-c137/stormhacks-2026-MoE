"""Live: the standup ingested through the API and ended by its host; real Gemini writes it up
(with the fallback chain) and real Gemini embeddings index it into an in-memory vector store.
Needs GEMINI_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM.
Deselected unless pytest runs with `-m live`."""

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
from contracts import AGENT_PARTICIPANT_ID, Report, Team

FIXTURES = Path(__file__).parent.parent / "fixtures"
WORKER_TOKEN = "live-worker-token"
settings = Settings()

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

    transcript = {s.text for s in standup.segments}
    members = {p.id for p in standup.members}
    assert report.summary
    assert report.tasks, "a standup with a stated refund commitment should yield a task"
    for task in report.tasks:
        assert task.quote in transcript
        assert task.owner_id is None or task.owner_id in members
        assert task.owner_id != AGENT_PARTICIPANT_ID
    assert all(d.quote in transcript for d in report.decisions)
    refund = [t for t in report.tasks if "refund" in t.title.lower()]
    assert refund and refund[0].owner_id == "p-bob"

    hits = await memory.search(team.id, "refund", k=10)
    for hit in hits:
        print(f"{hit.score:.3f} {hit.chunk.kind} {hit.chunk.text}")
    turns = [h.chunk for h in hits if h.chunk.kind == "transcript"]
    assert turns and turns[0].speaker_id == "p-bob"
    assert "refund the 14 affected users" in turns[0].text
