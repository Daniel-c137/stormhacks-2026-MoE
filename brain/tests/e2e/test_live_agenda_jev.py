"""Live: a standup tracked by real Jev, checked after every caption as the worker does with
JEV_MODEL set. Needs OPENROUTER_API_KEY and JEV_MODEL (e.g. typesafe/jev-1.13); Gemini is never
asked. Costs about $0.0005. Deselected unless pytest runs with `-m live`."""

import time
from datetime import UTC, datetime

import pytest
from api_support import ALEX, SARAH, TEAM

from brain.agent.timekeeping import JEV_SETTLE_S, track_agenda
from brain.config import Settings
from brain.llm import LLMUnavailable
from brain.llm.jev import JevClient, make_jev
from brain.store import InMemoryStore
from contracts import AGENT_PARTICIPANT_ID, Agenda, AgendaItem, TranscriptSegment

settings = Settings()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.openrouter_api_key and settings.jev_model),
        reason="set OPENROUTER_API_KEY and JEV_MODEL to run against Jev",
    ),
]

# (speaker, words, start, end): seconds from the meeting start
STANDUP = [
    (ALEX, "Morning. Quick one from me: the waitlist email draft is in the doc.", 0, 6),
    (SARAH, "I read it. The subject line is too long, and we should drop the second link.", 7, 14),
    (ALEX, "Fair. I'll shorten the subject and cut the link today.", 15, 20),
    (SARAH, "Then it goes out to the waitlist Thursday morning. That's settled.", 21, 26),
    (ALEX, "Did anyone see the game last night?", 27, 30),
    (SARAH, "On refunds: annual plans still have no answer.", 31, 36),
]


def no_gemini():
    raise LLMUnavailable("Gemini must not be asked while Jev keeps time")


@pytest.mark.anyio
async def test_live_jev_ticks_an_item_off_at_the_check_after_its_closing_line():
    jev = make_jev(settings)
    assert isinstance(jev, JevClient)
    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    meeting = await store.create_meeting(TEAM.id, "Standup", ALEX.id)
    titles = ["Waitlist email", "Refund policy", "Launch date"]
    await store.save_agenda(
        Agenda(
            meeting_id=meeting.id,
            items=[AgendaItem(id=f"i{n}", title=t) for n, t in enumerate(titles)],
            generated_at=datetime.now(UTC),
        )
    )

    states, seconds, cost = [], [], 0.0
    for n, (person, text, start, end) in enumerate(STANDUP):
        await store.add_segments(
            meeting.id,
            [
                TranscriptSegment(
                    seg_id=f"s{n}",
                    meeting_id=meeting.id,
                    speaker_id=person.id,
                    speaker_name=person.name,
                    text=text,
                    is_final=True,
                    t_start=start,
                    t_end=end,
                )
            ],
        )
        began = time.perf_counter()
        result = await track_agenda(store, no_gemini, meeting, end + JEV_SETTLE_S, jev=jev)
        seconds.append(time.perf_counter() - began)
        cost += jev.last_cost or 0
        items = {i.id: i for i in result.agenda.items}
        states.append((items["i0"].status, items["i1"].status, result.agenda.current_item_id))
        print(f"after line {n + 1}: {states[-1]} in {seconds[-1] * 1000:.0f} ms")
    print(f"{len(seconds)} checks, ${cost:.6f}")

    waitlist = items["i0"]
    assert [s[0] for s in states[:3]] == ["pending"] * 3  # still being weighed
    assert states[3][0] == "covered"  # ticked at the check right after "That's settled."
    assert (waitlist.covered_by, waitlist.covered_t) == (AGENT_PARTICIPANT_ID, 26)
    assert states[-1][1] == "pending" and states[-1][2] == "i1"  # refunds: still open, current
    assert max(seconds) < 3
