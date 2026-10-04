"""Questions about the code: the agent searches the team's repository, reads the top files at the
team's ref, and attaches the snippets its answer cites, copied from the file."""

import pytest
from api_support import ALEX, OTHER_TEAM, OUTSIDER, SARAH, TEAM
from ask_support import citing, evidence, ids_for, scripted
from conftest import MAIN_SHA, REFUNDS_PY, RELEASE_SHA

from brain.agent.ask import (
    BEGIN_DATA,
    END_DATA,
    CodeLines,
    DraftAnswer,
    PlannedCall,
    Question,
    ToolOrchestrator,
)
from brain.config import Settings
from brain.store import InMemoryStore
from contracts import GitHubSettings, JiraSettings, TeamSettings

pytestmark = pytest.mark.anyio

REFUNDS = "api/billing/refunds.py"
FEES = "api/billing/fees.py"
QUESTION = "What's the refund window in the code?"
WINDOW_LINE = REFUNDS_PY.splitlines().index("REFUND_WINDOW_DAYS = 30") + 1


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(teams=[TEAM, OTHER_TEAM], people=[ALEX, SARAH, OUTSIDER])


def question(text: str = QUESTION) -> Question:
    return Question(
        id="q-code",
        team_id=TEAM.id,
        text=text,
        asker_id=ALEX.id,
        asker_name=ALEX.name,
        visibility="public",
    )


async def configure(store, repo: str | None = "dropsubs/app", ref: str | None = None):
    await store.save_settings(
        TeamSettings(
            team_id=TEAM.id, github=GitHubSettings(repo=repo, ref=ref), jira=JiraSettings()
        )
    )


def orchestrator(llm, store, fake_github, **config) -> ToolOrchestrator:
    settings = Settings(_env_file=None, **({"github_mcp_url": "http://unused.invalid"} | config))
    return ToolOrchestrator(llm, store, settings=settings, github_target=fake_github.server)


def code(query: str = "refund window") -> PlannedCall:
    return PlannedCall(tool="github_code", query=query)


def file_lines(text: str, start: int, end: int) -> str:
    return "\n".join(text.splitlines()[start - 1 : end])


async def test_a_code_answer_attaches_the_cited_snippet_copied_from_the_file(store, fake_github):
    await configure(store)
    llm = scripted(code(), answer=citing(REFUNDS, text="Refunds are allowed for 30 days."))

    answer = await orchestrator(llm, store, fake_github).ask(question())

    (snippet,) = answer.snippets
    assert snippet.path == REFUNDS
    assert snippet.code == file_lines(REFUNDS_PY, snippet.start_line, snippet.end_line)
    assert snippet.start_line <= WINDOW_LINE <= snippet.end_line
    assert snippet.github_url == (
        f"https://github.com/dropsubs/app/blob/{MAIN_SHA}/{REFUNDS}"
        f"#L{snippet.start_line}-L{snippet.end_line}"
    )
    (source,) = answer.sources
    assert source.kind == "github_code"
    assert source.url == snippet.github_url
    assert REFUNDS in source.label
    assert answer.text == "Refunds are allowed for 30 days."
    assert answer.unavailable == []


async def test_only_cited_snippets_are_attached(store, fake_github):
    await configure(store)
    llm = scripted(code("billing"), answer=citing(FEES, text="We charge 30%."))

    answer = await orchestrator(llm, store, fake_github).ask(question("What fee do we charge?"))

    lines = evidence(llm.calls[1].prompt)
    assert any(REFUNDS in line for line in lines.values())  # found, shown, but not cited
    assert [s.path for s in answer.snippets] == [FEES]
    assert [s.url for s in answer.sources] == [answer.snippets[0].github_url]


async def test_an_uncited_code_answer_attaches_nothing(store, fake_github):
    await configure(store)
    llm = scripted(code(), answer=DraftAnswer(text="Thirty days.", evidence_ids=["e9"]))

    answer = await orchestrator(llm, store, fake_github).ask(question())

    assert answer.snippets == [] and answer.sources == []


async def test_the_answer_step_sees_numbered_lines_as_quoted_data(store, fake_github):
    await configure(store)
    llm = scripted(code(), answer=citing(REFUNDS))

    await orchestrator(llm, store, fake_github).ask(question())

    prompt = llm.calls[1].prompt
    fenced = prompt[prompt.index(BEGIN_DATA) : prompt.index(END_DATA)]
    assert f"{WINDOW_LINE:>4} | REFUND_WINDOW_DAYS = 30" in fenced
    assert "github_code" in llm.calls[0].prompt


async def test_the_model_picks_the_highlighted_lines_inside_the_snippet(store, fake_github):
    await configure(store)

    def answer(prompt: str) -> DraftAnswer:
        (e,) = ids_for(prompt, REFUNDS)
        return DraftAnswer(
            text="30 days.",
            evidence_ids=[e],
            code_lines=[CodeLines(evidence_id=e, start_line=WINDOW_LINE, end_line=WINDOW_LINE + 1)],
        )

    result = await orchestrator(scripted(code(), answer=answer), store, fake_github).ask(question())

    (snippet,) = result.snippets
    assert snippet.highlight == (WINDOW_LINE, WINDOW_LINE + 1)


async def test_invented_highlight_lines_never_change_the_code(store, fake_github):
    await configure(store)

    def answer(prompt: str) -> DraftAnswer:
        (e,) = ids_for(prompt, REFUNDS)
        return DraftAnswer(
            text="30 days.",
            evidence_ids=[e],
            code_lines=[
                CodeLines(evidence_id=e, start_line=400, end_line=420),
                CodeLines(evidence_id="e77", start_line=1, end_line=2),
            ],
        )

    result = await orchestrator(scripted(code(), answer=answer), store, fake_github).ask(question())

    (snippet,) = result.snippets
    assert snippet.code == file_lines(REFUNDS_PY, snippet.start_line, snippet.end_line)
    assert snippet.highlight == (WINDOW_LINE, WINDOW_LINE)  # the search match, not line 400


async def test_code_is_read_at_the_teams_ref(store, fake_github):
    await configure(store, ref="release")
    llm = scripted(code(), answer=citing(REFUNDS))

    answer = await orchestrator(llm, store, fake_github).ask(question())

    (snippet,) = answer.snippets
    assert "REFUND_WINDOW_DAYS = 14" in snippet.code
    assert f"/blob/{RELEASE_SHA}/" in snippet.github_url


async def test_unconfigured_github_lands_in_unavailable_without_code(store, fake_github):
    await configure(store)
    llm = scripted(code(), answer=citing(REFUNDS))

    answer = await orchestrator(llm, store, fake_github, github_mcp_url=None).ask(question())

    assert answer.snippets == [] and answer.sources == []
    assert any("GitHub" in u and "GITHUB_MCP_URL" in u for u in answer.unavailable)
    assert fake_github.calls == []
    assert len(llm.calls) == 1


async def test_a_team_without_a_repository_gets_no_code(store, fake_github):
    await configure(store, repo=None)
    llm = scripted(code(), answer=citing(REFUNDS))

    answer = await orchestrator(llm, store, fake_github).ask(question())

    assert answer.snippets == []
    assert any("no repository" in u for u in answer.unavailable)
    assert fake_github.calls == []


async def test_a_failed_read_lands_in_unavailable(store, fake_github):
    await configure(store, ref="no-such-branch")
    llm = scripted(code(), answer=citing(REFUNDS))

    answer = await orchestrator(llm, store, fake_github).ask(question())

    assert answer.snippets == []
    assert any(u.startswith("GitHub code failed") for u in answer.unavailable)


async def test_code_from_other_repositories_never_reaches_the_answer(store, fake_github):
    await configure(store)
    fake_github.ignore_scope = True
    llm = scripted(code("refund window days"), answer=citing(REFUNDS))

    answer = await orchestrator(llm, store, fake_github).ask(question())

    assert not any("secret" in call.prompt for call in llm.calls[1:])
    assert all("otherorg" not in (s.url or "") for s in answer.sources)
    reads = [args for tool, args in fake_github.calls if tool == "get_file_contents"]
    assert [a["path"] for a in reads] == [REFUNDS]
