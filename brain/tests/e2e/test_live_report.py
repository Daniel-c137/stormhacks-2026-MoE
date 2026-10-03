"""Live: the fixture transcript through real Gemini. Needs GEMINI_API_KEY and GEMINI_MODEL."""

from pathlib import Path

import pytest

from brain.config import Settings
from brain.llm import GeminiLLM, make_llm
from brain.report import TranscriptInput, build_report
from contracts import AGENT_PARTICIPANT_ID

FIXTURES = Path(__file__).parent.parent / "fixtures"
settings = Settings()

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.gemini_api_key and settings.gemini_model),
        reason="set GEMINI_API_KEY and GEMINI_MODEL to run against Gemini",
    ),
]


async def test_gemini_turns_the_standup_into_a_grounded_report():
    meeting = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    llm = make_llm(settings)
    assert isinstance(llm, GeminiLLM)

    report = await build_report(llm, meeting)

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
