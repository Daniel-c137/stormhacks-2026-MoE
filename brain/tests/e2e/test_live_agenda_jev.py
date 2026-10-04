"""Live: meetings tracked by real Jev, checked after every caption as the worker does with
JEV_MODEL set. Needs OPENROUTER_API_KEY and JEV_MODEL (e.g. typesafe/jev-1.13); Gemini is never
asked. Costs about $0.001. Deselected unless pytest runs with `-m live`."""

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


async def play(titles: list[str], lines) -> list[tuple]:
    """Each line saved and checked as it settles; after each, every item's (status, covered_t)
    and the item being discussed. Item ids are i0, i1, ... in the order of `titles`."""
    jev = make_jev(settings)
    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    meeting = await store.create_meeting(TEAM.id, "Standup", ALEX.id)
    await store.save_agenda(
        Agenda(
            meeting_id=meeting.id,
            items=[AgendaItem(id=f"i{n}", title=t) for n, t in enumerate(titles)],
            generated_at=datetime.now(UTC),
        )
    )
    after = []
    for n, (person, text, start, end) in enumerate(lines):
        segment = TranscriptSegment(
            seg_id=f"s{n}",
            meeting_id=meeting.id,
            speaker_id=person.id,
            speaker_name=person.name,
            text=text,
            is_final=True,
            t_start=start,
            t_end=end,
        )
        await store.add_segments(meeting.id, [segment])
        result = await track_agenda(store, no_gemini, meeting, end + JEV_SETTLE_S, jev=jev)
        after.append(
            (
                {i.id: (i.status, i.covered_t) for i in result.agenda.items},
                result.agenda.current_item_id,
            )
        )
        print(f"after line {n + 1}: {after[-1]}")
    return after


@pytest.mark.anyio
async def test_live_jev_ticks_an_item_when_its_result_is_reported():
    """Meeting 036aeb86 on 2026-10-04: the database check was reported done, and acknowledged,
    but sat under the threshold until the talk had moved on."""
    after = await play(
        ["Double-charge incident refunds and fixes", "Database availability double-check"],
        [
            (
                ALEX,
                "Um, so, Sarah, did you work on the double-charge incident refunds and fixes?",
                90.7,
                95.4,
            ),
            (SARAH, "Yes, I close the ticket.", 105.3, 106.5),
            (
                ALEX,
                "Uh, and also I did a double-check for, um, database availability. It seems like "
                "we have PostgreSQL and we can use it pretty much easily for our project, so "
                "that's good.",
                124.1,
                134.5,
            ),
        ],
    )

    assert after[0][0]["i0"] == ("pending", None)  # only asked about
    assert after[1][0]["i0"] == ("covered", 106.5)
    assert after[2][0]["i1"] == ("covered", 134.5)  # at the report, not later


@pytest.mark.anyio
async def test_live_jev_ticks_an_item_when_the_team_moves_on_to_work_off_the_agenda():
    after = await play(
        ["Refund policy", "Launch date"],
        [
            (ALEX, "On refunds: annual plans get a prorated refund within thirty days.", 0, 6),
            (SARAH, "And monthly plans get nothing back after the first week.", 7, 12),
            (ALEX, "Right, that matches what support has been telling people.", 13, 17),
            (
                SARAH,
                "Also, we need to choose a NoSQL database for all the unstructured event data.",
                19,
                25,
            ),
        ],
    )

    assert [a[0]["i0"][0] for a in after[:3]] == ["pending"] * 3
    assert after[3][0]["i0"] == ("covered", 17)  # the end of the last thing said about refunds
    assert after[3][1] is None  # the NoSQL talk is about no agenda item
