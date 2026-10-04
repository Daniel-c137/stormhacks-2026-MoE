"""Live: real Gemini plans a code search and answers from a file in a fake repository, citing a
snippet copied from that file. Needs GEMINI_API_KEY and GEMINI_MODEL. Deselected unless pytest
runs with `-m live`."""

import pytest
from api_support import ALEX, SARAH, TEAM
from conftest import MAIN_SHA, REFUNDS_PY, FakeGitHub

from brain.agent.ask import Question, ToolOrchestrator
from brain.config import Settings
from brain.llm import GeminiLLM, make_llm
from brain.store import InMemoryStore
from contracts import GitHubSettings, JiraSettings, TeamSettings

settings = Settings(openrouter_models=None)  # Gemini alone, never the fallback

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.gemini_api_key and settings.gemini_model),
        reason="set GEMINI_API_KEY and GEMINI_MODEL",
    ),
]


@pytest.mark.anyio
async def test_gemini_answers_the_refund_window_with_a_snippet_from_the_file():
    asker, team = SARAH, TEAM
    store = InMemoryStore(teams=[team], people=[ALEX, SARAH])
    await store.save_settings(
        TeamSettings(
            team_id=team.id, github=GitHubSettings(repo="dropsubs/app"), jira=JiraSettings()
        )
    )
    github = FakeGitHub()
    config = settings.model_copy(update={"github_mcp_url": "http://unused.invalid/github"})
    llm = make_llm(config)
    assert isinstance(llm, GeminiLLM)

    answer = await ToolOrchestrator(llm, store, settings=config, github_target=github.server).ask(
        Question(
            id="q-live-code",
            team_id=team.id,
            text="What's the refund window in the code?",
            asker_id=asker.id,
            asker_name=asker.name,
            visibility="public",
        )
    )

    print(f"answered by {llm.last_model}: {answer.text}")
    print(f"snippets: {answer.snippets}; unavailable: {answer.unavailable}")
    snippet = next(s for s in answer.snippets if s.path == "api/billing/refunds.py")
    lines = REFUNDS_PY.splitlines()[snippet.start_line - 1 : snippet.end_line]
    assert snippet.code == "\n".join(lines)
    assert "REFUND_WINDOW_DAYS = 30" in snippet.code
    assert snippet.github_url.startswith(
        f"https://github.com/dropsubs/app/blob/{MAIN_SHA}/api/billing/refunds.py#L"
    )
    assert any(s.kind == "github_code" and s.url == snippet.github_url for s in answer.sources)
    assert "30" in answer.text or "thirty" in answer.text.lower()  # answers read aloud
