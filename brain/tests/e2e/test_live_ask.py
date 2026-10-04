"""Live: the standup indexed with real Gemini embeddings, then a question planned and answered by
real Gemini. Needs GEMINI_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM.
Deselected unless pytest runs with `-m live`."""

from pathlib import Path

import pytest

from brain.agent.ask import Question, ToolOrchestrator
from brain.config import Settings
from brain.llm import GeminiEmbedder, GeminiLLM, make_embedder, make_llm
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.report import TranscriptInput
from brain.store import InMemoryStore
from contracts import Team

FIXTURES = Path(__file__).parent.parent / "fixtures"
settings = Settings()

pytestmark = [
    pytest.mark.live,
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


@pytest.mark.anyio
async def test_gemini_answers_the_waitlist_question_citing_alices_moment():
    standup = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    people = standup.members
    team = Team(id="t-live", name="Dropsubs", member_ids=[p.id for p in people])
    store = InMemoryStore(teams=[team], people=people)
    meeting = await store.create_meeting(team.id, standup.title, people[0].id)
    segments = [s.model_copy(update={"meeting_id": meeting.id}) for s in standup.segments]
    embedder = make_embedder(settings)
    assert isinstance(embedder, GeminiEmbedder)
    memory = MeetingMemory(embedder, InMemoryMemoryStore(dim=embedder.dim))
    await memory.index_meeting(team.id, meeting.id, segments)
    llm = make_llm(settings)
    assert isinstance(llm, GeminiLLM)
    bob = people[1]

    answer = await ToolOrchestrator(llm, store, settings=settings, memory=memory).ask(
        Question(
            id="q-live",
            team_id=team.id,
            text="What did we decide about the waitlist email?",
            asker_id=bob.id,
            asker_name=bob.name,
            visibility="public",
        )
    )

    print(f"answered by {llm.last_model}: {answer.text}")
    print(f"sources: {answer.sources}; unavailable: {answer.unavailable}")
    alice = next(s for s in segments if "waitlist email" in s.text)
    assert alice.speaker_name == "Alice Moreau"
    assert any(
        s.kind == "meeting" and s.meeting_id == meeting.id and s.t == alice.t_start
        for s in answer.sources
    )
    assert "0.9.4" in answer.text
