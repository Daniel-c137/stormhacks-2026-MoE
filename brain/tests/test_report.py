from datetime import date, datetime
from pathlib import Path

import pytest

from brain.llm import MockLLM
from brain.report import (
    ExtractedDecision,
    ExtractedLink,
    ExtractedRisk,
    ExtractedTask,
    ReportExtraction,
    TranscriptInput,
    build_report,
)
from contracts import AGENT_PARTICIPANT_ID, Report, TranscriptSegment, get_identity

pytestmark = pytest.mark.anyio

FIXTURES = Path(__file__).parent / "fixtures"


def standup(**changes) -> TranscriptInput:
    meeting = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    return meeting.model_copy(update=changes)


def segment(n: int) -> TranscriptSegment:
    return standup().segments[n - 1]


EXTRACTION = ReportExtraction(
    summary="Bob merged the fix and will refund users. The waitlist email waits for v0.9.4.",
    topics=["Double charge", " ", "Waitlist email"],
    decisions=[
        ExtractedDecision(text="Hold the waitlist email", made_by_id="p-alice", evidence=["s4"]),
        ExtractedDecision(text="Production stays on v0.9.3", made_by_id=None, evidence=["s3"]),
        ExtractedDecision(text="Invented decision", made_by_id="p-alice", evidence=[]),
    ],
    tasks=[
        ExtractedTask(
            title="Refund the users", owner_id="p-bob", due="2026-10-07", evidence=["s2"]
        ),
        ExtractedTask(
            title="Own the model retirement", owner_id="p-dave", due="2026-11-15", evidence=["s6"]
        ),
        ExtractedTask(title="Update DS-104", owner_id=AGENT_PARTICIPANT_ID, evidence=["s5"]),
        ExtractedTask(title="Ghost task", owner_id="p-bob", evidence=["s99"]),
        ExtractedTask(title="Backdated", owner_id="p-bob", due="2026-09-01", evidence=["s2"]),
        ExtractedTask(title="Vague date", owner_id="p-bob", due="next week", evidence=["s2"]),
    ],
    risks=[
        ExtractedRisk(text="Exploit still live", severity="high", evidence=["s3"]),
        ExtractedRisk(text="Unsupported risk", severity="low", evidence=[]),
    ],
    open_questions=["Who owns the model retirement?"],
    links=[
        ExtractedLink(kind="jira_issue", label="DS-104"),
        ExtractedLink(kind="jira_issue", label="DS-999"),
    ],
)


async def report_for(meeting: TranscriptInput | None = None) -> tuple[Report, MockLLM]:
    llm = MockLLM(structured={ReportExtraction: EXTRACTION})
    return await build_report(llm, meeting or standup()), llm


def task(report: Report, title: str):
    return next(t for t in report.tasks if t.title == title)


async def test_task_drafts_carry_owner_due_date_and_the_cited_moment():
    report, _ = await report_for()

    refund = task(report, "Refund the users")
    assert refund.id == "mtg-standup-task-1"
    assert refund.meeting_id == "mtg-standup"
    assert refund.owner_id == "p-bob"
    assert refund.due == date(2026, 10, 7)
    assert refund.quote == segment(2).text
    assert refund.t == segment(2).t_start
    assert (refund.include, refund.key, refund.jira_status) == (True, None, "draft")


async def test_an_owner_must_be_a_participant_and_never_the_agent():
    report, _ = await report_for()

    assert task(report, "Own the model retirement").owner_id is None
    assert task(report, "Update DS-104").owner_id is None


async def test_items_without_valid_evidence_are_dropped():
    report, _ = await report_for()

    assert "Ghost task" not in [t.title for t in report.tasks]
    assert "Invented decision" not in [d.text for d in report.decisions]
    assert [r.text for r in report.risks] == ["Exploit still live"]


async def test_due_dates_must_be_real_dates_on_or_after_the_meeting():
    report, _ = await report_for()

    assert task(report, "Own the model retirement").due == date(2026, 11, 15)
    assert task(report, "Backdated").due is None
    assert task(report, "Vague date").due is None


async def test_decisions_name_who_made_them_and_quote_the_transcript():
    report, _ = await report_for()

    hold, stays = report.decisions
    assert (hold.made_by, hold.t, hold.quote) == ("Alice Moreau", 24, segment(4).text)
    assert stays.made_by == "Carol Jensen"  # no participant given: whoever said the cited line
    assert hold.id == "mtg-standup-decision-1"


async def test_links_are_kept_only_when_the_transcript_mentions_them():
    report, _ = await report_for()

    assert [link.label for link in report.links] == ["DS-104"]


async def test_summary_sections_pass_through_without_blank_entries():
    report, _ = await report_for()

    assert report.summary.startswith("Bob merged the fix")
    assert report.topics == ["Double charge", "Waitlist email"]
    assert report.open_questions == ["Who owns the model retirement?"]
    assert report.blockers == []


async def test_the_prompt_gives_date_participants_and_labelled_segments():
    _, llm = await report_for()
    agent = get_identity().agent_name

    (call,) = llm.calls
    assert call.schema is ReportExtraction
    assert "Date: 2026-10-02" in call.prompt
    assert "- p-bob: Bob Okafor" in call.prompt
    assert f"- {AGENT_PARTICIPANT_ID}: {agent}" in call.prompt
    assert "[s2 00:06] Bob Okafor: The double-charge fix is merged." in call.prompt
    assert f"[s5 00:32] {agent}: DS-104" in call.prompt
    assert agent in call.system


async def test_only_final_segments_are_sent_once_each():
    meeting = standup()
    partial = segment(6).model_copy(
        update={"seg_id": "seg-6p", "text": "Someone nee", "is_final": False}
    )
    meeting = meeting.model_copy(update={"segments": [*meeting.segments, segment(2), partial]})

    _, llm = await report_for(meeting)

    prompt = llm.calls[0].prompt
    assert prompt.count("I'll refund the 14 affected users") == 1
    assert "Someone nee\n" not in prompt
    assert "[s6 " in prompt and "[s7 " not in prompt


async def test_participants_default_to_the_people_who_spoke():
    report, llm = await report_for(standup(members=[]))

    assert "- p-bob: Bob Okafor" in llm.calls[0].prompt
    assert task(report, "Refund the users").owner_id == "p-bob"


async def test_a_transcript_without_final_segments_is_rejected_before_calling_the_llm():
    meeting = standup(segments=[segment(1).model_copy(update={"is_final": False})])
    llm = MockLLM(structured={ReportExtraction: EXTRACTION})

    with pytest.raises(ValueError, match="no final segments"):
        await build_report(llm, meeting)
    assert llm.calls == []


async def test_without_a_meeting_date_due_dates_are_not_range_checked():
    report, llm = await report_for(standup(started_at=None))

    assert "Date: unknown" in llm.calls[0].prompt
    assert task(report, "Backdated").due == date(2026, 9, 1)


def test_the_fixture_meeting_is_a_friday():
    assert standup().started_at == datetime(2026, 10, 2, 9, 30)
    assert standup().started_at.strftime("%A") == "Friday"
