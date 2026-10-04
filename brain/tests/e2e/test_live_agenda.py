"""Live: a rough lobby topic rewritten, an agenda drafted, and standup stretches tracked, by real
Gemini. Needs GEMINI_API_KEY and GEMINI_MODEL (optionally GEMINI_FALLBACK_MODELS). Deselected
unless pytest runs with `-m live`."""

import re
from datetime import UTC, date, datetime

import pytest
from api_support import ALEX, SARAH, TEAM

from brain.agent.agenda import SuggestionInput, rewrite_topic, suggest_items, title_key
from brain.agent.timekeeping import track_agenda
from brain.config import Settings
from brain.llm import GeminiLLM, make_llm
from brain.store import InMemoryStore
from contracts import Agenda, AgendaItem, Source, TranscriptSegment

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
            person_id=ALEX.id,
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
    print(f"{result.agendas[0].model_dump_json(indent=1)} (answered by {llm.last_model})")

    waitlist = result.agendas[0].items[0]
    assert result.agendas[0].current_item_id == waitlist.id or waitlist.status == "covered"
    assert [i.status for i in result.agendas[0].items[1:]] == ["pending", "pending"]
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
            person_id=ALEX.id,
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
    print(f"{result.agendas[0].model_dump_json(indent=1)} (answered by {llm.last_model})")

    waitlist, refunds, launch = result.agendas[0].items
    assert launch.discussed_s == 0 and launch.status == "pending"
    assert result.agendas[0].current_item_id != launch.id
    assert waitlist.discussed_s > 0 or refunds.discussed_s > 0
    assert waitlist.discussed_s + refunds.discussed_s <= 55


@pytest.mark.anyio
async def test_live_gemini_drafts_items_that_fit_the_meeting_and_skip_what_is_on_the_agenda():
    llm = make_llm(settings)
    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    earlier = await store.create_meeting(TEAM.id, "Checkout sync", ALEX.id)
    meeting = await store.create_meeting(TEAM.id, "Weekly sync", ALEX.id, duration_min=30)
    existing = [AgendaItem(id="a1", title="Who owns the refund script?", minutes=10)]
    met = Source(kind="meeting", label="Checkout sync (2026-09-28)", meeting_id=earlier.id)
    jira = Source(kind="jira_issue", label="DS-117", url="https://example.test/browse/DS-117")
    inputs = [
        SuggestionInput("Open question", "Who owns the refund script?", met),
        SuggestionInput("Blocker", "Staging database is down since Friday", met),
        SuggestionInput("Risk (high)", "Queue backlog could delay password-reset emails", met),
        SuggestionInput("Superseded decision", "Keep background jobs on Postgres", met),
        SuggestionInput(
            "Decision contradicting an earlier one", "Move background jobs to Redis", met
        ),
        SuggestionInput("Overdue task", "Rotate the API keys (due 2026-09-01; DS-90, todo)", met),
        SuggestionInput(
            "Unfinished Jira issue",
            "DS-117: Refund the double-charged users (In Progress, Sarah Kim)",
            jira,
        ),
    ]

    items = await suggest_items(llm, meeting, inputs, date(2026, 10, 4), existing)
    for item in items:
        print(f"{item.minutes} min  {item.title}  <- {[s.label for s in item.sources]}")
    print(f"(answered by {llm.last_model})")

    assert 1 <= len(items) <= 6
    assert sum(i.minutes or 0 for i in items) <= 20
    assert all(i.minutes is None or 5 <= i.minutes <= 30 for i in items)
    for item in items:
        title = item.title.lower()
        assert item.sources and item.why
        assert len(item.title.split()) <= 12
        assert title_key(item.title) != title_key(existing[0].title)
        assert not ("refund" in title and "script" in title), item.title
        assert "checkout sync" not in title and "2026" not in title
        assert not re.search(r"\(\s*ds-\d+\s*\)\s*$", title), item.title
