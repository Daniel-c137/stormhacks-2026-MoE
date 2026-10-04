from pathlib import Path

import pytest

from brain.llm.mock import MockEmbedder
from brain.memory import Chunk, InMemoryMemoryStore, MeetingMemory, PgMemoryStore
from brain.report import TranscriptInput
from contracts import AGENT_PARTICIPANT_ID, Report, TaskDraft, get_identity

pytestmark = pytest.mark.anyio

FIXTURES = Path(__file__).parent / "fixtures"
STANDUP = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
REFUND_QUESTION = "Who will refund the affected users?"


def standup_report(*task_titles: str) -> Report:
    return Report(
        meeting_id=STANDUP.meeting_id,
        summary="Bob refunds double-charged users; the waitlist email waits for v0.9.4.",
        tasks=[
            TaskDraft(id=f"task-{i}", meeting_id=STANDUP.meeting_id, title=title)
            for i, title in enumerate(task_titles)
        ],
    )


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def memory(embedder) -> MeetingMemory:
    return MeetingMemory(embedder, InMemoryMemoryStore())


async def test_a_question_finds_the_turn_that_answers_it(memory):
    await memory.index_meeting("t-1", STANDUP.meeting_id, STANDUP.segments)

    hits = await memory.search("t-1", REFUND_QUESTION, k=3)

    top = hits[0].chunk
    assert (top.kind, top.speaker_id, top.meeting_id, top.t_start) == (
        "transcript",
        "p-bob",
        STANDUP.meeting_id,
        6,
    )
    assert len(hits) == 3


async def test_documents_and_queries_are_embedded_with_their_own_task(memory, embedder):
    await memory.index_meeting("t-1", STANDUP.meeting_id, STANDUP.segments, standup_report("x"))
    await memory.search("t-1", REFUND_QUESTION)

    (documents, document_task), (queries, query_task) = embedder.calls
    assert document_task == "document"
    assert len(documents) == 5 + 2  # five people's turns, the summary and one task
    assert (queries, query_task) == ([REFUND_QUESTION], "query")


async def test_search_is_scoped_to_the_team(memory):
    await memory.index_meeting("t-2", STANDUP.meeting_id, STANDUP.segments)

    assert await memory.search("t-1", REFUND_QUESTION) == []


async def test_indexing_a_meeting_again_replaces_what_it_had(memory):
    await memory.index_meeting(
        "t-1", STANDUP.meeting_id, STANDUP.segments, standup_report("Refund users", "Old task")
    )
    await memory.index_meeting(
        "t-1", STANDUP.meeting_id, STANDUP.segments, standup_report("Refund users")
    )

    hits = await memory.search("t-1", "task", k=50)
    assert [h.chunk.text for h in hits if h.chunk.kind == "task"] == ["Task: Refund users"]
    assert len(hits) == 5 + 2


async def test_deleting_the_transcript_keeps_the_report(memory):
    await memory.index_meeting(
        "t-1", STANDUP.meeting_id, STANDUP.segments, standup_report("Refund users")
    )

    await memory.delete_meeting_transcript(STANDUP.meeting_id)

    hits = await memory.search("t-1", REFUND_QUESTION, k=50)
    assert sorted(h.chunk.kind for h in hits) == ["summary", "task"]


async def test_a_blank_question_finds_nothing_without_embedding(memory, embedder):
    assert await memory.search("t-1", "   ") == []
    assert embedder.calls == []


async def test_a_report_for_another_meeting_is_refused(memory):
    other = Report(meeting_id="mtg-other", summary="elsewhere")

    with pytest.raises(ValueError, match="mtg-other"):
        await memory.index_meeting("t-1", STANDUP.meeting_id, STANDUP.segments, other)


def test_the_embedder_must_fit_the_store():
    with pytest.raises(ValueError, match="768"):
        MeetingMemory(MockEmbedder(dim=64), InMemoryMemoryStore(dim=768))


async def test_the_standup_round_trips_through_postgres(memory_pool, embedder):
    memory = MeetingMemory(embedder, PgMemoryStore(memory_pool, dim=embedder.dim))
    await memory.index_meeting(
        "t-1", STANDUP.meeting_id, STANDUP.segments, standup_report("Own the model retirement")
    )

    hits = await memory.search("t-1", REFUND_QUESTION, k=3)

    assert hits[0].chunk.speaker_id == "p-bob"
    assert await memory.search("t-2", REFUND_QUESTION) == []


AGENTS_LINE = "DS-104 is still In Progress in Jira, though the fix is merged."


@pytest.fixture(params=["in_memory", "postgres"])
def either_memory(request, embedder) -> MeetingMemory:
    if request.param == "in_memory":
        return MeetingMemory(embedder, InMemoryMemoryStore())
    pool = request.getfixturevalue("memory_pool")
    return MeetingMemory(embedder, PgMemoryStore(pool, dim=embedder.dim))


async def test_the_agents_own_words_are_not_indexed(either_memory):
    await either_memory.index_meeting(
        "t-1", STANDUP.meeting_id, STANDUP.segments, standup_report("Own the model retirement")
    )

    hits = await either_memory.search("t-1", AGENTS_LINE, k=50)

    assert hits
    assert all(h.chunk.speaker_id != AGENT_PARTICIPANT_ID for h in hits)
    assert not any("In Progress" in h.chunk.text for h in hits)


async def test_indexing_again_drops_agent_chunks_indexed_before(either_memory, embedder):
    stale = Chunk(
        id=f"{STANDUP.meeting_id}:transcript:seg-5",
        team_id="t-1",
        meeting_id=STANDUP.meeting_id,
        kind="transcript",
        text=f"{get_identity().agent_name}: {AGENTS_LINE}",
        speaker_id=AGENT_PARTICIPANT_ID,
        t_start=32,
        t_end=36,
    )
    await either_memory.store.replace_meeting(
        STANDUP.meeting_id, [stale], await embedder.embed([stale.text])
    )

    await either_memory.index_meeting("t-1", STANDUP.meeting_id, STANDUP.segments)

    hits = await either_memory.search("t-1", AGENTS_LINE, k=50)
    assert len(hits) == 5
    assert all(h.chunk.speaker_id != AGENT_PARTICIPANT_ID for h in hits)
