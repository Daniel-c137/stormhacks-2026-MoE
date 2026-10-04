"""Dates people and prompts see are in the team's time zone; stored times stay in UTC.

A meeting on Saturday evening in Vancouver starts at 01:30 UTC on Sunday. For a team on
America/Vancouver it is dated Saturday 3 October in ask evidence, in "Today", in decision links
and in due dates. A team on UTC sees what it saw before. Every test runs on both stores."""

from datetime import UTC, date, datetime, timedelta

import pytest
from api_support import ALEX, OTHER_TEAM, OUTSIDER, SARAH, TEAM, postgres_world
from ask_support import evidence, scripted
from fact_check_support import (
    CONNECTED,
    CONTRADICTED,
    PR_41,
    RELEASED,
    live,
    merged_after_release,
    say,
)

from brain.agent.agenda import store_inputs
from brain.agent.ask import PlannedCall, Question, ToolOrchestrator
from brain.agent.factcheck import FactChecker, FactCheckPlan, FactCheckVerdicts
from brain.agent.pipeline import ReportPipeline
from brain.config import Settings
from brain.llm import MockLLM
from brain.report import ExtractedDecision, ExtractedTask, ReportExtraction
from brain.report.decisions import DecisionVerdicts
from brain.store import InMemoryStore, Store
from contracts import (
    Decision,
    GitHubSettings,
    JiraSettings,
    Report,
    TaskDraft,
    TeamSettings,
    TranscriptSegment,
)

pytestmark = pytest.mark.anyio

VANCOUVER = "America/Vancouver"
STARTED = datetime(2026, 10, 4, 1, 30, tzinfo=UTC)  # Saturday 3 October, 18:30 in Vancouver
ASKED = STARTED + timedelta(minutes=30)

# (team time zone, the day the meeting and the question fall on, its weekday)
ZONES = [
    pytest.param(VANCOUVER, date(2026, 10, 3), "Saturday", id="vancouver"),
    pytest.param("UTC", date(2026, 10, 4), "Sunday", id="utc"),
]


@pytest.fixture(params=["memory", "postgres"])
async def store(request, anyio_backend) -> Store:
    if request.param == "postgres":
        return await postgres_world(request.getfixturevalue("pg_dsn"))
    return InMemoryStore(teams=[TEAM, OTHER_TEAM], people=[ALEX, SARAH, OUTSIDER])


@pytest.fixture
def clock(monkeypatch):
    """The brain's clock reads ASKED: 02:00 UTC on Sunday, 19:00 on Saturday in Vancouver."""
    monkeypatch.setattr("brain.zones.now", lambda: ASKED)


async def in_zone(store: Store, zone: str) -> None:
    await store.save_settings(
        TeamSettings(
            team_id=TEAM.id,
            github=GitHubSettings(repo="dropsubs/app"),
            jira=JiraSettings(project="DS"),
            timezone=zone,
        )
    )


async def meeting_at(store: Store, title: str, start: datetime, *, status: str = "live"):
    meeting = await store.create_meeting(
        TEAM.id, title, ALEX.id, scheduled_start=start, duration_min=30
    )
    meeting = await store.start_meeting(meeting.id, start)
    if status == "processing":
        end = start + timedelta(minutes=30)
        meeting = await store.transition_status(meeting.id, {"live"}, "processing", at=end)
    return meeting


def question(text: str) -> Question:
    return Question(
        id="q-1",
        team_id=TEAM.id,
        text=text,
        asker_id=ALEX.id,
        asker_name=ALEX.name,
        visibility="public",
    )


def orchestrator(llm, store) -> ToolOrchestrator:
    return ToolOrchestrator(llm, store, settings=Settings(_env_file=None))


HOLD = "Hold the waitlist email until v0.9.4 is out"


async def standup_with_a_decision(store: Store):
    meeting = await meeting_at(store, "Friday standup", STARTED)
    await store.save_report(
        Report(
            meeting_id=meeting.id,
            summary="The waitlist email waits for v0.9.4.",
            decisions=[
                Decision(
                    id=f"{meeting.id}-decision-1",
                    meeting_id=meeting.id,
                    text=HOLD,
                    made_by="Alice Moreau",
                    t=24,
                    quote="Let's hold the waitlist email until v0.9.4 is out.",
                )
            ],
        )
    )
    return meeting


# ask


@pytest.mark.parametrize(("zone", "day", "weekday"), ZONES)
async def test_ask_evidence_dates_the_meeting_by_the_teams_day(store, zone, day, weekday):
    await in_zone(store, zone)
    await standup_with_a_decision(store)
    llm = scripted(
        PlannedCall(tool="decisions", query="waitlist"), PlannedCall(tool="recent_meetings")
    )

    await orchestrator(llm, store).ask(question("When did we decide on the waitlist email?"))

    lines = list(evidence(llm.calls[1].prompt).values())
    assert len(lines) == 2
    assert all(f"({day.isoformat()})" in line for line in lines), lines
    other = day + timedelta(days=1 if zone == VANCOUVER else -1)
    assert not any(other.isoformat() in line for line in lines), lines


@pytest.mark.parametrize(("zone", "day", "weekday"), ZONES)
async def test_today_is_the_teams_day_when_the_question_is_asked(store, clock, zone, day, weekday):
    await in_zone(store, zone)
    llm = scripted()

    await orchestrator(llm, store).ask(question("What is due today?"))

    assert f"Today: {day.isoformat()} ({weekday})" in llm.calls[0].prompt


@pytest.mark.parametrize(("zone", "day", "weekday"), ZONES)
async def test_a_task_due_on_the_teams_today_is_not_overdue_yet(store, clock, zone, day, weekday):
    await in_zone(store, zone)
    meeting = await meeting_at(store, "Planning", STARTED - timedelta(days=7))
    saturday = date(2026, 10, 3)
    await store.save_report(
        Report(
            meeting_id=meeting.id,
            summary="s",
            tasks=[
                TaskDraft(id="t-1", meeting_id=meeting.id, title="Send the email", due=saturday)
            ],
        )
    )
    llm = scripted(PlannedCall(tool="tasks", status="overdue"))

    await orchestrator(llm, store).ask(question("What is overdue?"))

    overdue = len(llm.calls) == 2 and "Send the email" in llm.calls[1].prompt
    assert overdue == (day > saturday)


# fact-checks


@pytest.mark.parametrize(("zone", "day", "weekday"), ZONES)
async def test_fact_checks_see_the_teams_today_and_github_dates(
    store, clock, fake_jira, fake_github, zone, day, weekday
):
    merged_after_release(fake_github)
    # 03:00 UTC on 1 October is the evening of 30 September in Vancouver.
    fake_github.pulls[41] = PR_41 | {"merged_at": "2026-10-01T03:00:00Z"}
    meeting = await live(store, timezone=zone)
    await say(store, meeting, (SARAH, RELEASED, 10))
    llm = MockLLM(
        structured={
            FactCheckPlan: CONTRADICTED[0],
            FactCheckVerdicts: FactCheckVerdicts(checks=[]),
        }
    )
    checker = FactChecker(
        llm,
        store,
        settings=Settings(_env_file=None, **CONNECTED),
        jira_target=fake_jira.server,
        github_target=fake_github.server,
    )

    await checker.tick(meeting, 60)

    plan_prompt, verdict_prompt = (call.prompt for call in llm.calls)
    assert f"Today: {day.isoformat()} ({weekday})" in plan_prompt
    merged = "2026-09-30" if zone == VANCOUVER else "2026-10-01"
    assert f"merged {merged}" in verdict_prompt


# agenda suggestions


@pytest.mark.parametrize(("zone", "day", "weekday"), ZONES)
async def test_agenda_inputs_name_earlier_meetings_by_the_teams_day(store, zone, day, weekday):
    from brain.zones import team_zone

    await in_zone(store, zone)
    earlier = await meeting_at(store, "Checkout sync", STARTED)
    await store.save_report(
        Report(meeting_id=earlier.id, summary="s", open_questions=["Who owns refunds?"])
    )
    upcoming = await store.create_meeting(TEAM.id, "Next sync", ALEX.id)

    [found] = await store_inputs(store, TEAM.id, upcoming.id, day, team_zone(zone))

    assert found.source.label == f"Checkout sync ({day.isoformat()})"


# the write-up


async def finished_meeting(store: Store, start: datetime, line: str):
    meeting = await meeting_at(store, "Release sync", start)
    await store.add_segments(
        meeting.id,
        [
            TranscriptSegment(
                seg_id=f"{meeting.id}-1",
                meeting_id=meeting.id,
                speaker_id=SARAH.id,
                speaker_name=SARAH.name,
                text=line,
                is_final=True,
                t_start=0,
                t_end=4,
            )
        ],
    )
    end = start + timedelta(minutes=30)
    return await store.transition_status(meeting.id, {"live"}, "processing", at=end)


@pytest.mark.parametrize(("zone", "day", "weekday"), ZONES)
async def test_decision_links_date_past_meetings_by_the_teams_day(store, zone, day, weekday):
    await in_zone(store, zone)
    await standup_with_a_decision(store)
    meeting = await finished_meeting(
        store, STARTED + timedelta(days=7), "Send the waitlist email now, before v0.9.4 is out."
    )
    llm = MockLLM(
        structured={
            ReportExtraction: ReportExtraction(
                summary="Send now.",
                decisions=[
                    ExtractedDecision(
                        text="Send the waitlist email before v0.9.4 is out", evidence=["s1"]
                    )
                ],
            ),
            DecisionVerdicts: DecisionVerdicts(),
        }
    )

    await ReportPipeline(store, lambda: llm).run(meeting.id)

    [link_prompt] = [c.prompt for c in llm.calls if c.schema is DecisionVerdicts]
    assert f"({day.isoformat()}, Alice Moreau) {HOLD}" in link_prompt
    assert f"({(day + timedelta(days=7)).isoformat()}, " in link_prompt  # the new meeting


@pytest.mark.parametrize(("zone", "day", "weekday"), ZONES)
async def test_due_dates_resolve_against_the_meetings_day_in_the_teams_zone(
    store, zone, day, weekday
):
    await in_zone(store, zone)
    meeting = await finished_meeting(store, STARTED, "I'll send the waitlist email by tonight.")
    llm = MockLLM(
        structured={
            ReportExtraction: ReportExtraction(
                summary="Sarah sends the email.",
                tasks=[
                    ExtractedTask(title="Send it tonight", due="2026-10-03", evidence=["s1"]),
                    ExtractedTask(title="Send it Friday", due="2026-10-02", evidence=["s1"]),
                    ExtractedTask(title="Send it Sunday", due="2026-10-04", evidence=["s1"]),
                ],
            )
        }
    )

    report = await ReportPipeline(store, lambda: llm).run(meeting.id)

    assert f"Date: {day.isoformat()} ({weekday})" in llm.calls[0].prompt
    due = {t.title: t.due for t in report.tasks}
    assert due["Send it Friday"] is None  # before the meeting, in any zone
    assert due["Send it Sunday"] == date(2026, 10, 4)
    # On the meeting's own day only where the meeting is on Saturday.
    assert due["Send it tonight"] == (date(2026, 10, 3) if zone == VANCOUVER else None)
    saved = await store.report(meeting.id)
    assert [t.due for t in saved.tasks] == [t.due for t in report.tasks]
    assert (await store.meeting(meeting.id)).started_at == STARTED  # stored times stay UTC
