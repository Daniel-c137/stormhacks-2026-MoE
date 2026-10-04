"""Live: a rough lobby topic rewritten, and standup stretches tracked, by real Gemini. Needs
GEMINI_API_KEY and GEMINI_MODEL (optionally GEMINI_FALLBACK_MODELS). Deselected unless pytest runs
with `-m live`."""

from datetime import UTC, datetime

import pytest
from api_support import ALEX, SARAH, TEAM

from brain.agent.agenda import rewrite_topic
from brain.agent.timekeeping import track_agenda
from brain.config import Settings
from brain.llm import GeminiLLM, make_llm
from brain.store import InMemoryStore
from contracts import Agenda, AgendaItem, TranscriptSegment

settings = Settings(openrouter_models=None)  # Gemini alone, never the fallback

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.gemini_api_key and settings.gemini_model),
        reason="set GEMINI_API_KEY and GEMINI_MODEL to run against Gemini",
    ),
]


@pytest.mark.anyio
async def test_live_gemini_rewrites_redis_vs_postgres_into_one_agenda_item():
    llm = make_llm(settings)
    assert isinstance(llm, GeminiLLM)

    title = await rewrite_topic(llm, "redis vs postgres?")
    print(f"{title!r} (answered by {llm.last_model})")

    assert title and "\n" not in title
    assert len(title) <= 80 and len(title.split()) <= 12
    assert any(word in title.lower() for word in ("queue", "redis", "postgres"))


@pytest.mark.anyio
async def test_live_gemini_tracks_a_standup_stretch_about_the_waitlist_email():
    llm = make_llm(settings)
    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    meeting = await store.create_meeting(TEAM.id, "Standup", ALEX.id)
    titles = ["Waitlist email", "Refund policy", "Launch date"]
    await store.save_agenda(
        Agenda(
            meeting_id=meeting.id,
            items=[AgendaItem(id=f"i{n}", title=t, minutes=5) for n, t in enumerate(titles)],
            generated_at=datetime.now(UTC),
        )
    )
    lines = [
        (ALEX, "Morning. Quick one from me: the waitlist email draft is in the doc."),
        (SARAH, "I read it. The subject line is too long, and we should drop the second link."),
        (ALEX, "Fair. I'll shorten the subject and cut the link today."),
        (SARAH, "Then it can go out to the waitlist Thursday morning."),
        (ALEX, "Great, waitlist email goes Thursday. That's settled."),
    ]
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
                t_start=n * 8.0,
                t_end=n * 8.0 + 7,
            )
            for n, (person, text) in enumerate(lines)
        ],
    )

    result = await track_agenda(store, lambda: llm, meeting, now=60)
    print(f"{result.agenda.model_dump_json(indent=1)} (answered by {llm.last_model})")

    waitlist = result.agenda.items[0]
    assert result.agenda.current_item_id == waitlist.id or waitlist.status == "covered"
    assert [i.status for i in result.agenda.items[1:]] == ["pending", "pending"]
    assert result.nudges == []


@pytest.mark.anyio
async def test_live_gemini_splits_a_two_topic_standup_and_gives_an_unmentioned_item_no_time():
    llm = make_llm(settings)
    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    meeting = await store.create_meeting(TEAM.id, "Standup", ALEX.id)
    titles = ["Waitlist email", "Refunds for the double charge", "Launch date"]
    await store.save_agenda(
        Agenda(
            meeting_id=meeting.id,
            items=[AgendaItem(id=f"i{n}", title=t, minutes=5) for n, t in enumerate(titles)],
            generated_at=datetime.now(UTC),
        )
    )
    lines = [
        (ALEX, "Morning. Quick one from me: the waitlist email draft is in the doc."),
        (SARAH, "I read it. The subject line is too long, and we should drop the second link."),
        (ALEX, "Fair. I'll fix both and send the waitlist email Thursday morning."),
        (SARAH, "Next thing: twelve customers were charged twice on Monday."),
        (ALEX, "Stripe shows the duplicate payments. I can refund all twelve today."),
        (SARAH, "Do it, and send each of them a short apology with the refund."),
        (ALEX, "Will do. The refunds go out this afternoon."),
    ]
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
                t_start=n * 8.0,
                t_end=n * 8.0 + 7,
            )
            for n, (person, text) in enumerate(lines)
        ],
    )

    result = await track_agenda(store, lambda: llm, meeting, now=70)
    print(f"{result.agenda.model_dump_json(indent=1)} (answered by {llm.last_model})")

    waitlist, refunds, launch = result.agenda.items
    assert launch.discussed_s == 0 and launch.status == "pending"
    assert result.agenda.current_item_id != launch.id
    assert waitlist.discussed_s > 0 or refunds.discussed_s > 0
    assert waitlist.discussed_s + refunds.discussed_s <= 55
