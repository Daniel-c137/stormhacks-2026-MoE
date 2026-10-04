"""Live: real Gemini fact-checks "the refund fix from PR 41 is already released" against a GitHub
(FakeGitHub) where PR #41 was merged after the latest release. Needs GEMINI_API_KEY and
GEMINI_MODEL (optionally GEMINI_FALLBACK_MODELS). Deselected unless pytest runs with `-m live`."""

import pytest
from api_support import ALEX, SARAH, TEAM
from fact_check_support import merged_after_release, say

from brain.agent.factcheck import FactChecker
from brain.config import Settings
from brain.llm import GeminiLLM, make_llm
from brain.store import InMemoryStore
from contracts import GitHubSettings, JiraSettings, TeamSettings

settings = Settings(openrouter_models=None)  # Gemini alone, never the fallback

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.gemini_api_key and settings.gemini_model),
        reason="set GEMINI_API_KEY and GEMINI_MODEL to run against Gemini",
    ),
]


@pytest.mark.anyio
async def test_live_gemini_finds_pr_41_merged_but_not_released(fake_github):
    merged_after_release(fake_github)
    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    await store.save_settings(
        TeamSettings(
            team_id=TEAM.id, github=GitHubSettings(repo="dropsubs/app"), jira=JiraSettings()
        )
    )
    meeting = await store.create_meeting(TEAM.id, "Refund sync", ALEX.id)
    await say(
        store,
        meeting,
        (ALEX, "Where are we on the double charge?", 5),
        (SARAH, "The refund fix from PR 41 is already released.", 12),
    )
    llm = make_llm(settings)
    assert isinstance(llm, GeminiLLM)
    # Only the fake GitHub: no real Jira or meeting memory.
    checker = FactChecker(
        llm,
        store,
        settings=settings.model_copy(
            update={"github_mcp_url": "http://unused.invalid/github", "jira_mcp_url": None}
        ),
        github_target=fake_github.server,
    )

    response = await checker.tick(meeting, 60)

    print(f"checked by {llm.last_model}: {response.model_dump_json(indent=2)}")
    print(f"GitHub calls: {fake_github.calls}")
    [fact] = response.checks
    assert fact.verdict == "contradicted"
    assert (fact.speaker_name, fact.recipient_id) == (SARAH.name, SARAH.id)
    assert fact.finding, "the check says what the records show"
    assert any(s.kind in ("github_pr", "github_release") for s in fact.sources)
    assert fake_github.comments == []
