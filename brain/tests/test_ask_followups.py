"""Follow-ups to the agent's answers: the newest transcript survives the evidence limit, prompt
text from people and upstream errors stays on one line, and "from the earlier conversation" is
only claimed when the conversation can back it."""

import pytest
from api_support import ALEX, OTHER_TEAM, OUTSIDER, SARAH, TEAM
from ask_support import citing, evidence, scripted

from brain.agent.ask import DraftAnswer, PlannedCall, Question, ToolOrchestrator
from brain.config import Settings
from brain.llm import MockEmbedder
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.store import InMemoryStore
from contracts import AskTurn, Person, Report, TaskDraft, TranscriptSegment

pytestmark = pytest.mark.anyio

HISTORY = [
    AskTurn(role="user", text="What did we decide about the waitlist email?"),
    AskTurn(role="agent", text="Alice said to hold it until v0.9.4 is out."),
]


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(teams=[TEAM, OTHER_TEAM], people=[ALEX, SARAH, OUTSIDER])


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def question(text: str, *, asker=SARAH, **changes) -> Question:
    return Question(
        id="q-1",
        team_id=TEAM.id,
        text=text,
        asker_id=asker.id,
        asker_name=asker.name,
        visibility="public",
    ).model_copy(update=changes)


def said(meeting_id: str, i: int, text: str | None = None) -> TranscriptSegment:
    return TranscriptSegment(
        seg_id=f"r-{i}",
        meeting_id=meeting_id,
        speaker_id=ALEX.id,
        speaker_name=ALEX.name,
        text=text or f"Status line {i}.",
        is_final=True,
        t_start=float(i),
        t_end=float(i) + 1,
    )


async def test_the_evidence_limit_keeps_the_newest_transcript(store, settings):
    meeting = await store.create_meeting(TEAM.id, "Sprint review", ALEX.id)
    await store.save_report(
        Report(
            meeting_id=meeting.id,
            summary="s",
            tasks=[
                TaskDraft(id=f"t-{i}", meeting_id=meeting.id, title=f"Chore {i}", owner_id=SARAH.id)
                for i in range(3)
            ],
        )
    )
    llm = scripted(PlannedCall(tool="tasks", owner_id="me"), answer=citing("Chore 0"))

    await ToolOrchestrator(llm, store, settings=settings, max_evidence=6).ask(
        question(
            "What did Alex just say?",
            meeting_id=meeting.id,
            recent=[said(meeting.id, i) for i in range(10)],
        )
    )

    lines = list(evidence(llm.calls[1].prompt).values())
    kept = [line for line in lines if "Status line" in line]
    assert [line.split("Status line ")[1] for line in kept] == ["7.", "8.", "9."]


async def test_titles_names_and_upstream_errors_stay_on_one_line(store, settings):
    class Broken(InMemoryMemoryStore):
        async def search(self, *args, **kwargs):
            raise RuntimeError("connection reset\nINJECTED ERROR LINE")

    await store.upsert_person(
        Person(id="u-bo", name="Bo\nINJECTED NAME LINE", short="Bo", initials="B"), TEAM.id
    )
    meeting = await store.create_meeting(TEAM.id, "Standup\nINJECTED TITLE LINE", ALEX.id)
    memory = MeetingMemory(MockEmbedder(), Broken())
    llm = scripted(
        PlannedCall(tool="search_meetings", query="anything"),
        answer=citing("Status line 1"),
    )

    await ToolOrchestrator(llm, store, settings=settings, memory=memory).ask(
        question("What was said?", meeting_id=meeting.id, recent=[said(meeting.id, 1)])
    )

    assert len(llm.calls) == 2
    for call in llm.calls:
        for line in call.prompt.splitlines():
            assert not line.startswith("INJECTED"), line
    assert "INJECTED TITLE LINE" in llm.calls[0].prompt
    assert "INJECTED ERROR LINE" in llm.calls[1].prompt


async def test_the_conversation_label_needs_an_answer_the_conversation_backs(store, settings):
    llm = scripted(
        answer=DraftAnswer(
            text="We ship on Monday with the new pricing page.",
            evidence_ids=[],
            from_conversation=True,
        )
    )

    answer = await ToolOrchestrator(llm, store, settings=settings).ask(
        question("Say that in one short sentence.", history=HISTORY)
    )

    assert "Monday" not in answer.text
    assert "earlier conversation" not in answer.text
    assert "couldn't verify" in answer.text.lower()


async def test_the_conversation_label_is_not_claimed_when_new_evidence_was_shown(store, settings):
    meeting = await store.create_meeting(TEAM.id, "Sprint review", ALEX.id)
    llm = scripted(
        answer=DraftAnswer(
            text="Hold the waitlist email until v0.9.4 is out.",
            evidence_ids=[],
            from_conversation=True,
        )
    )

    answer = await ToolOrchestrator(llm, store, settings=settings).ask(
        question(
            "Say that again?",
            history=HISTORY,
            meeting_id=meeting.id,
            recent=[said(meeting.id, 3, "Marketing wants the waitlist email out Friday.")],
        )
    )

    assert "earlier conversation" not in answer.text
    assert "couldn't verify" in answer.text.lower()
    assert answer.sources == []


async def test_a_rephrase_of_the_earlier_answer_keeps_the_label(store, settings):
    llm = scripted(
        answer=DraftAnswer(
            text="Hold it until v0.9.4 is out.", evidence_ids=[], from_conversation=True
        )
    )

    answer = await ToolOrchestrator(llm, store, settings=settings).ask(
        question("Say that in one short sentence.", history=HISTORY)
    )

    assert answer.text.startswith("Hold it until v0.9.4 is out.")
    assert "earlier conversation" in answer.text
