"""Live: the standup indexed with real Gemini embeddings, then a question planned and answered by
real Gemini. Needs GEMINI_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM.
Deselected unless pytest runs with `-m live`."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.agent.ask import FROM_CONVERSATION, NO_EVIDENCE, UNVERIFIED, Question, ToolOrchestrator
from brain.config import Settings
from brain.llm import GeminiEmbedder, GeminiLLM, make_embedder, make_llm
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.report import TranscriptInput
from brain.store import InMemoryStore
from contracts import Agenda, AgendaItem, AskTurn, Source, Team, get_identity

FIXTURES = Path(__file__).parent.parent / "fixtures"
settings = Settings(openrouter_models=None)  # Gemini alone, never the fallback

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


async def indexed_standup():
    """The standup as a meeting of a live team, stored and indexed with real Gemini embeddings."""
    standup = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    people = standup.members
    team = Team(id="t-live", name="Dropsubs", member_ids=[p.id for p in people])
    store = InMemoryStore(teams=[team], people=people)
    meeting = await store.create_meeting(team.id, standup.title, people[0].id)
    segments = [s.model_copy(update={"meeting_id": meeting.id}) for s in standup.segments]
    await store.add_segments(meeting.id, segments)
    embedder = make_embedder(settings)
    assert isinstance(embedder, GeminiEmbedder)
    memory = MeetingMemory(embedder, InMemoryMemoryStore(dim=embedder.dim))
    await memory.index_meeting(team.id, meeting.id, segments)
    llm = make_llm(settings)
    assert isinstance(llm, GeminiLLM)
    return team, people, meeting, segments, store, memory, llm


@pytest.mark.anyio
async def test_gemini_answers_the_waitlist_question_citing_alices_moment():
    team, people, meeting, segments, store, memory, llm = await indexed_standup()
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


@pytest.mark.anyio
async def test_gemini_keeps_the_version_as_written_in_the_meeting():
    """#70: asked in the meeting, the answer keeps the version as the evidence writes it."""
    team, people, meeting, segments, store, memory, llm = await indexed_standup()
    bob = people[1]

    answer = await ToolOrchestrator(llm, store, settings=settings, memory=memory).ask(
        Question(
            id="q-live-version",
            team_id=team.id,
            text="What did we decide about the waitlist email?",
            asker_id=bob.id,
            asker_name=bob.name,
            visibility="public",
            meeting_id=meeting.id,
            recent=segments,
        )
    )

    print(f"answered by {llm.last_model}: {answer.text}")
    print(f"sources: {answer.sources}; unavailable: {answer.unavailable}")
    assert "0.9.4" in answer.text
    assert "zero point" not in answer.text.casefold()


@pytest.mark.anyio
async def test_gemini_answers_a_home_follow_up_without_meta_phrases():
    team, people, meeting, segments, store, memory, llm = await indexed_standup()
    bob = people[1]
    history = [
        AskTurn(role="user", text="What did we decide about the waitlist email, and when?"),
        AskTurn(
            role="agent",
            text="During the Friday standup on 2026-10-04, the team decided to hold the waitlist "
            "email until v0.9.4 is out. Alice Moreau proposed this decision during the meeting.",
        ),
    ]

    answer = await ToolOrchestrator(llm, store, settings=settings, memory=memory).ask(
        Question(
            id="q-live-follow-up",
            team_id=team.id,
            text="Who said that?",
            asker_id=bob.id,
            asker_name=bob.name,
            visibility="private",
            history=history,
        )
    )

    print(f"answered by {llm.last_model}: {answer.text}")
    print(f"sources: {answer.sources}; unavailable: {answer.unavailable}")
    assert answer.text not in (UNVERIFIED, NO_EVIDENCE)
    assert "Alice" in answer.text
    said = answer.text.removesuffix(FROM_CONVERSATION)
    assert get_identity().agent_name not in said
    alice = next(s for s in segments if "waitlist email" in s.text)
    cited = any(s.meeting_id == meeting.id and s.t == alice.t_start for s in answer.sources)
    assert cited or answer.text.endswith(FROM_CONVERSATION)


@pytest.mark.anyio
async def test_gemini_says_whats_left_on_the_agenda():
    """#93: asked in a meeting with a three-item agenda, one item covered, the answer names the
    two items left and cites the agenda."""
    standup = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    people = standup.members
    team = Team(id="t-live", name="Dropsubs", member_ids=[p.id for p in people])
    store = InMemoryStore(teams=[team], people=people)
    meeting = await store.create_meeting(team.id, "Sprint review", people[0].id)
    bob = people[1]  # asks about their own agenda
    await store.save_agenda(
        Agenda(
            meeting_id=meeting.id,
            person_id=bob.id,
            items=[
                AgendaItem(
                    id="i-1",
                    title="Release checklist",
                    status="covered",
                    minutes=10,
                    discussed_s=540,
                ),
                AgendaItem(id="i-2", title="Billing bug triage", minutes=5, discussed_s=130),
                AgendaItem(id="i-3", title="Hiring plan", minutes=15),
            ],
            generated_at=datetime.now(UTC),
            current_item_id="i-2",
            tracked_until=700.0,
        )
    )
    llm = make_llm(settings)
    assert isinstance(llm, GeminiLLM)

    answer = await ToolOrchestrator(llm, store, settings=settings).ask(
        Question(
            id="q-live-agenda",
            team_id=team.id,
            text="What's left on the agenda?",
            asker_id=bob.id,
            asker_name=bob.name,
            visibility="public",
            meeting_id=meeting.id,
        )
    )

    print(f"answered by {llm.last_model}: {answer.text}")
    print(f"sources: {answer.sources}; unavailable: {answer.unavailable}")
    assert Source(kind="meeting", label="Agenda", meeting_id=meeting.id) in answer.sources
    assert "billing bug triage" in answer.text.casefold()
    assert "hiring plan" in answer.text.casefold()
