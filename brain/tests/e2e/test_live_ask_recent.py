"""Live (#108): a question asked in a noisy meeting. Fragments and the name called alone around it
are neither evidence nor something to complete; the answer rests on what was actually said.
Needs GEMINI_API_KEY and GEMINI_MODEL (the OpenRouter fallback, when configured, covers quota).
Deselected unless pytest runs with `-m live`."""

import pytest

from brain.agent.ask import Question, ToolOrchestrator
from brain.config import Settings
from brain.llm import make_llm
from brain.store import InMemoryStore
from contracts import Person, Team, TranscriptSegment, get_identity

settings = Settings()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.gemini_api_key and settings.gemini_model),
        reason="set GEMINI_API_KEY and GEMINI_MODEL",
    ),
]

DANIAL = Person(id="p-danial", name="Danial", short="Danial", initials="D")
REZA = Person(id="p-reza", name="Mohammad Reza", short="Reza", initials="MR")
HOSSEIN = Person(id="p-hossein", name="Hossein", short="Hossein", initials="H")


def said(meeting_id: str, who: Person, text: str, t: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(
        seg_id=f"s-{t}",
        meeting_id=meeting_id,
        speaker_id=who.id,
        speaker_name=who.name,
        text=text,
        is_final=True,
        t_start=t,
        t_end=end,
    )


@pytest.mark.anyio
async def test_a_question_in_a_noisy_meeting_is_answered_from_what_was_said():
    agent = get_identity().agent_name
    people = [DANIAL, REZA, HOSSEIN]
    team = Team(id="t-live", name="Dropsubs", member_ids=[p.id for p in people])
    store = InMemoryStore(teams=[team], people=people)
    meeting = await store.create_meeting(team.id, "Billing sync", DANIAL.id)
    m = meeting.id
    recent = [
        said(m, HOSSEIN, "We will hold the waitlist email until v0.9.4 ships.", 80.0, 84.0),
        said(m, REZA, "Oh, I need-", 90.2, 91.0),
        said(m, DANIAL, f"Um, {agent},", 112.5, 114.8),
        said(m, DANIAL, "یه سؤال این الان چرا فقط از...", 137.9, 140.5),
        said(m, REZA, "Uh, w- b- b- b- b", 140.6, 142.0),
        said(m, DANIAL, f"{agent}, what did Hossein say about the waitlist email?", 160.0, 163.0),
    ]

    answer = await ToolOrchestrator(make_llm(settings), store, settings=settings).ask(
        Question(
            id="q-live-noisy",
            team_id=team.id,
            text="what did Hossein say about the waitlist email?",
            asker_id=DANIAL.id,
            asker_name=DANIAL.name,
            visibility="public",
            meeting_id=m,
            recent=recent,
        )
    )

    print(f"answer: {answer.text}")
    print(f"sources: {answer.sources}; unavailable: {answer.unavailable}")
    assert "0.9.4" in answer.text
    assert {s.t for s in answer.sources if s.meeting_id == m} <= {80, 160}
    assert any(s.t == 80 for s in answer.sources)
    assert "need" not in answer.text.casefold()
