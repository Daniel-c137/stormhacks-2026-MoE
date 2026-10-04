"""Live (#115): someone joins seven minutes into a meeting with a live agenda and gets a short
private catch-up: where the meeting is now, what was decided and what was asked of them.
Needs GEMINI_API_KEY and GEMINI_MODEL (the OpenRouter fallback, when configured, covers quota).
Deselected unless pytest runs with `-m live`."""

from datetime import UTC, datetime, timedelta

import pytest

from brain.agent.catchup import catch_up
from brain.config import Settings
from brain.llm import make_llm
from brain.store import InMemoryStore
from contracts import (
    Agenda,
    AgendaItem,
    CatchUpRequest,
    Person,
    Team,
    TranscriptSegment,
    get_identity,
)

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
SARA = Person(id="p-sara", name="Sara Ahmadi", short="Sara", initials="SA")

# Seven minutes of a weekly sync, before Sara joins at 7:05.
SAID = [
    (DANIAL, "Okay, let's get going. First up is the refund double charge.", 12),
    (REZA, "So the double charge only hits annual plans that were upgraded mid-cycle.", 31),
    (REZA, "PR 41 fixes it by charging the refund once against the original invoice.", 52),
    (HOSSEIN, "I tested it on staging yesterday and the totals match now.", 78),
    (DANIAL, "Do we ship it this week or wait for the release train?", 101),
    (REZA, "I'd rather not wait, support is getting three tickets a day about it.", 118),
    (HOSSEIN, "Agreed. The risk is low, it's one code path.", 140),
    (DANIAL, "Okay, decided: we ship the refund fix on Friday as a hotfix.", 158),
    (DANIAL, "Reza, you own the hotfix and the release note.", 175),
    (REZA, "Sure, I'll have the note ready Thursday.", 189),
    (DANIAL, "Next item, the pricing page launch.", 214),
    (HOSSEIN, "The new pricing page is done except for the annual plan banner.", 236),
    (HOSSEIN, "The banner still says twenty percent off and it should say fifteen.", 255),
    (DANIAL, "Sara wrote that copy, so she should check it.", 281),
    (DANIAL, "Sara, when you get here, can you review the banner copy before Wednesday?", 297),
    (REZA, "We also need the pricing page behind the feature flag until then.", 330),
    (HOSSEIN, "The flag is already in, it's called pricing v2.", 352),
    (DANIAL, "Good. Let's keep going on pricing, there's the FAQ section too.", 380),
    (HOSSEIN, "The FAQ still mentions the old refund window of fourteen days.", 401),
]


def segments(meeting_id: str) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            seg_id=f"s-{t}",
            meeting_id=meeting_id,
            speaker_id=who.id,
            speaker_name=who.name,
            text=text,
            is_final=True,
            t_start=t,
            t_end=t + 6,
        )
        for who, text, t in SAID
    ]


@pytest.mark.anyio
async def test_a_late_joiner_is_caught_up_on_the_decision_and_the_current_item():
    agent = get_identity().agent_name
    people = [DANIAL, REZA, HOSSEIN, SARA]
    team = Team(id="t-live", name="Dropsubs", member_ids=[p.id for p in people])
    store = InMemoryStore(teams=[team], people=people)
    meeting = await store.create_meeting(team.id, "Dropsubs weekly", DANIAL.id)
    meeting = await store.update_meeting(
        meeting.model_copy(update={"started_at": datetime.now(UTC) - timedelta(seconds=425)})
    )
    for person in people:
        meeting = await store.add_participant(meeting.id, person.id)
    await store.add_segments(meeting.id, segments(meeting.id))
    await store.save_agenda(
        Agenda(
            meeting_id=meeting.id,
            items=[
                AgendaItem(id="i-1", title="Refund double charge", status="covered", minutes=5),
                AgendaItem(id="i-2", title="Pricing page launch", minutes=10),
                AgendaItem(id="i-3", title="Hiring update", minutes=5),
            ],
            generated_at=datetime.now(UTC),
            current_item_id="i-2",
            tracked_until=400,
        )
    )

    caught = await catch_up(
        store,
        lambda: make_llm(settings),
        meeting,
        CatchUpRequest(participant_id=SARA.id, since=0, until=425),
    )

    print(f"catch-up:\n{caught.text}")
    print(f"source times: {caught.source_times}")
    assert caught.text is not None
    lines = caught.text.splitlines()
    assert "Pricing page launch" in lines[1]  # where the meeting is now
    decided = [line for line in lines if line.startswith("Decided:")]
    assert decided and "Friday" in " ".join(decided)
    assert any("banner" in line.casefold() for line in lines if line.startswith("For you:"))
    assert agent in lines[-1]
    assert len(caught.text.split()) <= 120
    assert caught.source_times and all(0 <= t <= 425 for t in caught.source_times)
