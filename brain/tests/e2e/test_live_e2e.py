"""Live: the standup fixture through real Gemini, then approved drafts to a Jira MCP server over
HTTP. Needs GEMINI_API_KEY and GEMINI_MODEL (optionally GEMINI_FALLBACK_MODELS); Jira is the
local FakeJira. Deselected unless pytest runs with `-m live`."""

from pathlib import Path

import pytest

from brain.cli import main
from brain.config import Settings
from brain.llm import GeminiLLM, make_llm
from brain.report import ProcessedMeeting, TranscriptInput, build_report
from contracts import AGENT_PARTICIPANT_ID

FIXTURES = Path(__file__).parent.parent / "fixtures"
settings = Settings(openrouter_models=None)  # Gemini alone, never the fallback

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.gemini_api_key and settings.gemini_model),
        reason="set GEMINI_API_KEY and GEMINI_MODEL to run against Gemini",
    ),
]


@pytest.mark.anyio
async def test_gemini_turns_the_standup_into_a_grounded_report():
    meeting = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    llm = make_llm(settings)
    assert isinstance(llm, GeminiLLM)

    report = await build_report(llm, meeting)
    print(f"answered by {llm.last_model} (chain: {', '.join(llm.models)})")
    assert llm.last_model in llm.models

    transcript = {s.text for s in meeting.segments}
    members = {p.id for p in meeting.members}
    assert report.summary
    assert report.tasks, "a standup with a stated refund commitment should yield a task"
    for task in report.tasks:
        assert task.quote in transcript
        assert task.owner_id is None or task.owner_id in members
        assert task.owner_id != AGENT_PARTICIPANT_ID
    refund = [t for t in report.tasks if "refund" in t.title.lower()]
    assert refund and refund[0].owner_id == "p-bob"
    assert all(d.quote in transcript for d in report.decisions)


def test_gemini_report_to_jira_issues(jira_env, tmp_path):
    review_file = tmp_path / "standup.review.json"

    assert main(["report", str(FIXTURES / "standup.json"), "--out", str(review_file)]) == 0
    drafts = ProcessedMeeting.model_validate_json(review_file.read_text()).report.tasks
    assert drafts and jira_env.created == []

    assert main(["push", str(review_file), "--all", "--approved-by", "Alice Moreau"]) == 0

    pushed = ProcessedMeeting.model_validate_json(review_file.read_text()).report.tasks
    assert [c["summary"] for c in jira_env.created] == [t.title for t in drafts]
    assert all(t.key and t.jira_status == "todo" for t in pushed)
