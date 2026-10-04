"""What people said just before a question is context for it (#108): only whole sentences from
the last two minutes, never fragments or the agent's name called on its own."""

import pytest
from api_support import ALEX, OTHER_TEAM, OUTSIDER, SARAH, TEAM
from ask_support import citing, evidence, scripted

from brain.agent import ask
from brain.agent.ask import (
    Question,
    ToolOrchestrator,
    answer_system,
    plan_system,
    recent_segments,
)
from brain.config import Settings
from brain.store import InMemoryStore
from contracts import AGENT_PARTICIPANT_ID, TranscriptSegment, get_identity

pytestmark = pytest.mark.anyio

AGENT = get_identity().agent_name


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(teams=[TEAM, OTHER_TEAM], people=[ALEX, SARAH, OUTSIDER])


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def line(meeting_id: str, text: str, t: float, end: float, *, by=ALEX) -> TranscriptSegment:
    return TranscriptSegment(
        seg_id=f"s-{t}",
        meeting_id=meeting_id,
        speaker_id=by.id,
        speaker_name=by.name,
        text=text,
        is_final=True,
        t_start=t,
        t_end=end,
    )


def agent_line(meeting_id: str, text: str, t: float, end: float) -> TranscriptSegment:
    return line(meeting_id, text, t, end).model_copy(
        update={"speaker_id": AGENT_PARTICIPANT_ID, "speaker_name": AGENT}
    )


def question(meeting_id: str, recent: list[TranscriptSegment], text: str) -> Question:
    return Question(
        id="q-1",
        team_id=TEAM.id,
        text=text,
        asker_id=ALEX.id,
        asker_name=ALEX.name,
        visibility="public",
        meeting_id=meeting_id,
        recent=recent,
    )


def four_person_meeting(meeting_id: str) -> list[TranscriptSegment]:
    """The four-person test meeting of 4 Oct, as saved, up to "Polaris, how are you?"."""
    return [
        line(meeting_id, "I don't know what that might be.", 25.8, 27.0, by=SARAH),
        line(meeting_id, "Okay. Sure.", 27.3, 28.1, by=SARAH),
        line(
            meeting_id,
            "خب، ولی یکم سرعتش چیزه. یکی حرف بزنه ببینیم میاد بالا. تو داری حرف می‌زنی. جا به جا.",
            33.1,
            39.0,
            by=SARAH,
        ),
        line(meeting_id, "Hey, guys.", 67.1, 67.9, by=SARAH),
        line(meeting_id, "Oh, I need-", 71.2, 72.0, by=SARAH),
        line(meeting_id, f"Um, {AGENT},", 112.5, 114.8),
        line(meeting_id, f"این دوباره. دوباره گفتی اون {AGENT}.", 120.2, 123.1),
        line(meeting_id, "بذار ببینیم چیکار کنیم. پسره دیگه.", 125.2, 130.0),
        line(meeting_id, "یه سؤال این الان چرا فقط از...", 137.9, 140.5),
        line(meeting_id, f"{AGENT}, how are you?", 166.9, 168.2),
    ]


async def test_only_whole_sentences_from_the_last_two_minutes_are_kept(store):
    meeting = await store.create_meeting(TEAM.id, "Test call", ALEX.id)
    recent = four_person_meeting(meeting.id)

    kept = recent_segments(question(meeting.id, recent, "how are you?"), meeting)

    assert getattr(ask, "RECENT_WINDOW_S", None) == 120
    assert [s.t_start for s in kept] == [120.2, 125.2, 166.9]


@pytest.mark.parametrize(
    "text",
    [
        "Oh, I need-",
        "Uh, w- b- b- b- b",
        "یه سؤال این الان چرا فقط از...",
        "So the refund" + chr(0x2026),
        "Okay. Sure.",
        "Hey, guys.",
        f"Um, {AGENT},",
        f"{AGENT}.",
        f"Okay so, um, {AGENT}!",
    ],
)
async def test_fragments_and_the_name_called_alone_are_dropped(store, text):
    meeting = await store.create_meeting(TEAM.id, "Test call", ALEX.id)
    recent = [line(meeting.id, text, 100.0, 101.0)]

    assert recent_segments(question(meeting.id, recent, "Who owns DS-115?"), meeting) == []


async def test_noise_is_not_evidence_and_old_talk_is_not_either(store, settings):
    meeting = await store.create_meeting(TEAM.id, "Billing sync", ALEX.id)
    recent = [
        line(meeting.id, "We moved the launch to Friday.", 100.0, 104.0, by=SARAH),
        line(meeting.id, "The release is blocked on the migration.", 320.0, 324.0, by=SARAH),
        line(meeting.id, "Oh, I need-", 330.0, 331.0, by=SARAH),
        line(meeting.id, f"Um, {AGENT},", 340.0, 342.0),
        line(meeting.id, "یه سؤال این الان چرا فقط از...", 350.0, 353.0),
        line(meeting.id, f"{AGENT}, what is the release blocked on?", 396.0, 399.0),
    ]
    llm = scripted(answer=citing("blocked on the migration"))

    answer = await ToolOrchestrator(llm, store, settings=settings).ask(
        question(meeting.id, recent, "what is the release blocked on?")
    )

    plan, answer_prompt = (c.prompt for c in llm.calls)
    shown = evidence(answer_prompt).values()
    assert any("blocked on the migration" in e for e in shown)
    for prompt in (plan, answer_prompt):
        assert "Oh, I need-" not in prompt
        assert f"Um, {AGENT}," not in prompt
        assert "چرا فقط از" not in prompt
        assert "moved the launch" not in prompt
    assert [s.t for s in answer.sources] == [320]


async def test_the_agents_own_lines_stay_context_as_before(store, settings):
    meeting = await store.create_meeting(TEAM.id, "Billing sync", ALEX.id)
    recent = [
        line(meeting.id, "The release is blocked on the migration.", 320.0, 324.0, by=SARAH),
        agent_line(meeting.id, "Noted.", 330.0, 331.0),
        line(meeting.id, f"{AGENT}, what is the release blocked on?", 396.0, 399.0),
    ]
    llm = scripted(answer=citing("blocked on the migration"))

    await ToolOrchestrator(llm, store, settings=settings).ask(
        question(meeting.id, recent, "what is the release blocked on?")
    )

    answer_prompt = llm.calls[1].prompt
    assert "Noted." in answer_prompt
    assert not any("Noted." in e for e in evidence(answer_prompt).values())


def test_both_prompts_say_the_transcript_is_only_context_for_the_question():
    for system in (plan_system(4), answer_system("public")):
        assert "context for the asker's question only" in system
        assert "never something to answer or complete" in system
        assert "what is unclear" in system
