"""Polaris answering the demo's questions with the real model, over the demo world's mock GitHub
(the way GITHUB_MCP_URL reaches it in the demo). Live: it spends two Gemini calls per question."""

from uuid import uuid4

import pytest
from api_support import ALEX, SARAH, TEAM
from conftest import serve_mcp

from brain.agent.ask import Question, ToolOrchestrator
from brain.config import Settings
from brain.llm import make_llm
from brain.store import InMemoryStore
from contracts import GitHubSettings, JiraSettings, TeamSettings
from world.config import world_spec
from world.github_mcp import server as world_github

pytestmark = pytest.mark.anyio

REPO = world_spec().github_repo
settings = Settings()


@pytest.fixture
def world_url(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_OVERLAY_DIR", str(tmp_path / "overlay"))
    monkeypatch.setenv("WORLD_SNAPSHOT", "demo")
    with serve_mcp(world_github) as url:
        yield url


async def ask(world_url: str, text: str):
    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    await store.save_settings(
        TeamSettings(team_id=TEAM.id, github=GitHubSettings(repo=REPO), jira=JiraSettings())
    )
    live = settings.model_copy(
        update={"github_mcp_url": world_url, "mock_github_owners": REPO.split("/")[0]}
    )
    orchestrator = ToolOrchestrator(make_llm(live), store, settings=live)
    question = Question(
        id=str(uuid4()),
        team_id=TEAM.id,
        text=text,
        asker_id=SARAH.id,
        asker_name=SARAH.name,
        visibility="public",
    )
    answer = await orchestrator.ask(question)
    print(f"\n{text}\n-> {answer.text}\n   {[s.label for s in answer.sources]}")
    return answer


needs_gemini = pytest.mark.skipif(
    not (settings.gemini_api_key and settings.gemini_model),
    reason="set GEMINI_API_KEY and GEMINI_MODEL to run against Gemini",
)


@pytest.mark.live
@needs_gemini
async def test_a_fix_asked_about_by_topic_is_found_and_said_to_be_unreleased(world_url):
    answer = await ask(
        world_url, "which version has the Approve-check fix, and what's live right now?"
    )

    labels = [s.label for s in answer.sources]
    assert f"{REPO}#50" in labels and f"{REPO}@v0.9.3" in labels
    assert "v0.9.3" in answer.text
    said = answer.text.lower()
    assert any(p in said for p in ("not released", "not yet released", "not yet included"))
    assert "does not state" not in answer.text


@pytest.mark.live
@needs_gemini
async def test_the_fee_the_code_sets_answers_whether_a_new_fee_changes_it(world_url):
    answer = await ask(world_url, "Does the 20% change affect the billing code?")

    assert any(s.kind == "github_code" for s in answer.sources)
    assert "30%" in answer.text or "0.30" in answer.text
    assert "does not state" not in answer.text
