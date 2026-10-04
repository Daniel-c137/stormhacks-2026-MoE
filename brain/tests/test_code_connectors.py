"""Several GitHub repositories and GitLab projects behind every read path: Ask's tools, code
search, fact-check evidence and keyterms. Citations name the repository: owner/name#41 for
GitHub, group/project#7 and group/project!12 (merge requests) for GitLab.

The fakes answer like the official servers and only for the repository a call names, so a
result in the wrong repository would show."""

from typing import Any

import pytest
from api_support import ALEX, SARAH, TEAM
from ask_support import citing, evidence, scripted
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, EmbeddedResource, TextContent, TextResourceContents

from brain.agent.ask import AskPlan, PlannedCall, Question, ToolOrchestrator
from brain.agent.code import code_evidence_across
from brain.agent.factcheck import (
    ClaimPlan,
    ClaimVerdict,
    FactChecker,
    FactCheckPlan,
    FactCheckVerdicts,
)
from brain.agent.team_tools import TeamToolbox, github_readers, gitlab_readers, pick
from brain.config import Settings
from brain.github import GitHubReader
from brain.gitlab import GitLabError, GitLabReader, web_address
from brain.keyterms import meeting_keyterms
from brain.llm import MockLLM
from brain.store import InMemoryStore
from contracts import (
    CodeRepo,
    GitHubSettings,
    GitLabSettings,
    Source,
    TeamSettings,
    TranscriptSegment,
)

pytestmark = pytest.mark.anyio

APP, WEB_REPO = "dropsubs/app", "dropsubs/web"
INFRA = "dropsubs/platform/infra"
GITLAB = "https://gitlab.example.com"
SHA = "a" * 40
GL_SHA = "b" * 40


# fakes


class RepoGitHub:
    """GitHub's MCP server over several repositories: each call answers only for its owner/repo
    (code search for its repo: qualifier)."""

    def __init__(self):
        self.repos: dict[str, dict[str, Any]] = {
            APP: {
                "issues": {41: "Inbox sync drops a page of emails"},
                "pulls": {50: "Enforce the Approve check in the cancel service"},
                "releases": [{"tag_name": "v0.9.3", "published_at": "2026-09-30T22:00:00Z"}],
                "files": {"api/billing/fees.py": "# fees\nFEE_RATE = 0.30\n"},
            },
            WEB_REPO: {
                "issues": {
                    41: "Pricing page still says 30%",
                    7: "Landing page says the fee is 30%",
                },
                "pulls": {8: "Landing page: 20% success fee"},
                "releases": [],
                "files": {"lib/pricing.ts": "// pricing\nexport const SUCCESS_FEE_PERCENT = 20;\n"},
            },
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.broken: set[str] = set()
        self.server = MCPServer("github")

        def repo_of(owner: str, repo: str) -> dict[str, Any]:
            name = f"{owner}/{repo}"
            if name in self.broken:
                raise ToolError(f"502 Bad Gateway for {name}")
            if name not in self.repos:
                raise ToolError("404 Not Found")
            return self.repos[name]

        def item(name: str, number: int, title: str, kind: str) -> dict[str, Any]:
            path = "pull" if kind == "pr" else "issues"
            return {
                "number": number,
                "title": title,
                "state": "open",
                "html_url": f"https://github.com/{name}/{path}/{number}",
            }

        def search(kind: str, query: str, owner: str, repo: str) -> dict[str, Any]:
            data = repo_of(owner, repo)
            words = query.lower().split()
            found = [
                item(f"{owner}/{repo}", n, title, kind)
                for n, title in data["pulls" if kind == "pr" else "issues"].items()
                if all(w in title.lower() for w in words)
            ]
            return {"total_count": len(found), "items": found}

        @self.server.tool()
        def search_issues(query: str, owner: str, repo: str) -> dict[str, Any]:
            self.calls.append(("search_issues", {"repo": f"{owner}/{repo}", "query": query}))
            return search("issue", query, owner, repo)

        @self.server.tool()
        def search_pull_requests(query: str, owner: str, repo: str) -> dict[str, Any]:
            self.calls.append(("search_pull_requests", {"repo": f"{owner}/{repo}"}))
            return search("pr", query, owner, repo)

        @self.server.tool()
        def issue_read(method: str, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
            self.calls.append(("issue_read", {"repo": f"{owner}/{repo}", "n": issue_number}))
            issues = repo_of(owner, repo)["issues"]
            if issue_number not in issues:
                raise ToolError("404 Not Found")
            return item(f"{owner}/{repo}", issue_number, issues[issue_number], "issue")

        @self.server.tool()
        def pull_request_read(method: str, owner: str, repo: str, pullNumber: int) -> dict:
            pulls = repo_of(owner, repo)["pulls"]
            if pullNumber not in pulls:
                raise ToolError("404 Not Found")
            return item(f"{owner}/{repo}", pullNumber, pulls[pullNumber], "pr")

        @self.server.tool()
        def list_releases(
            owner: str, repo: str, perPage: int | None = None
        ) -> list[dict[str, Any]]:
            self.calls.append(("list_releases", {"repo": f"{owner}/{repo}"}))
            return repo_of(owner, repo)["releases"]

        @self.server.tool()
        def search_code(query: str, perPage: int | None = None) -> dict[str, Any]:
            words = [w.lower() for w in query.split() if ":" not in w]
            name = next(w.split(":", 1)[1] for w in query.split() if w.startswith("repo:"))
            self.calls.append(("search_code", {"repo": name}))
            files = repo_of(*name.split("/"))["files"]
            items = [
                {"name": p.rsplit("/", 1)[-1], "path": p, "repository": name}
                for p, text in files.items()
                if all(w in (p + text).lower() for w in words)
            ]
            return {"total_count": len(items), "items": items}

        @self.server.tool()
        def get_file_contents(owner: str, repo: str, path: str, ref: str | None = None):
            text = repo_of(owner, repo)["files"][path]
            return CallToolResult(
                content=[
                    TextContent(type="text", text="successfully downloaded text file"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=f"repo://{owner}/{repo}/sha/{SHA}/contents/{path}", text=text
                        ),
                    ),
                ]
            )


class FakeGitLab:
    """GitLab's MCP server (19.5 tool names) over one project; anything else is 404."""

    def __init__(self):
        self.project = INFRA
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.diff_fails = False
        web = f"{GITLAB}/{INFRA}"
        self.issues = {
            5: {
                "iid": 5,
                "title": "Load test before the waitlist opens",
                "state": "opened",
                "description": "Replay 2,500 first syncs against staging.",
                "web_url": f"{web}/-/issues/5",
            }
        }
        self.mrs = {
            4: {
                "iid": 4,
                "title": "Status page: check the API and the sync workers every minute",
                "state": "merged",
                "merged_at": "2026-09-28T23:00:00Z",
                "description": "Closes #3",
                "web_url": f"{web}/-/merge_requests/4",
            }
        }
        self.files = {"monitoring/status.yml": "# status page\ninterval_seconds: 60\n"}
        self.elsewhere = {
            "iid": 9,
            "title": "Load test secrets",
            "state": "opened",
            "web_url": f"{GITLAB}/other/secret/-/issues/9",
        }
        self.server = MCPServer("gitlab")

        def ours(project_id: str) -> None:
            if project_id != self.project:
                raise ToolError("404 Project Not Found")

        @self.server.tool()
        def search(scope: str, search: str, project_id: str, per_page: int = 20) -> CallToolResult:
            self.calls.append(("search", {"scope": scope, "search": search}))
            ours(project_id)
            words = search.lower().split()
            if scope == "blobs":
                found = [
                    {"path": p, "data": t, "startline": 1, "project_id": 61840213}
                    for p, t in self.files.items()
                    if all(w in (p + t).lower() for w in words)
                ]
            else:
                records = self.mrs if scope == "merge_requests" else self.issues
                found = [r for r in records.values() if all(w in r["title"].lower() for w in words)]
                found.append(self.elsewhere)  # a misbehaving server: dropped by the reader
            return CallToolResult(content=[TextContent(type="text", text=json(found))])

        @self.server.tool()
        def get_work_item(project_id: str, work_item_iid: int) -> dict[str, Any]:
            ours(project_id)
            if work_item_iid not in self.issues:
                raise ToolError(f"404 Work item #{work_item_iid} Not Found")
            raw = self.issues[work_item_iid]
            # GraphQL's shape: the iid as a string, state in capitals, webUrl
            return {
                "iid": str(raw["iid"]),
                "title": raw["title"],
                "state": "OPEN",
                "description": raw["description"],
                "webUrl": raw["web_url"],
            }

        @self.server.tool()
        def get_merge_request(
            project_id: str,
            merge_request_iid: int,
            include: list[str] | None = None,
            detail: str | None = None,
            diffs_first: int | None = None,
        ) -> dict[str, Any]:
            self.calls.append(("get_merge_request", {"iid": merge_request_iid, "include": include}))
            ours(project_id)
            if merge_request_iid not in self.mrs:
                raise ToolError("404 Merge request Not Found")
            result = dict(self.mrs[merge_request_iid])
            if include == ["diffs"]:
                if self.diff_fails:
                    raise ToolError("diffs are too large")
                assert detail == "full_patch"
                result["diffs"] = [
                    {
                        "old_path": "monitoring/status.yml",
                        "new_path": "monitoring/status.yml",
                        "new_file": True,
                        "diff": "@@ -0,0 +1,2 @@\n+# status page\n+interval_seconds: 60\n",
                    }
                ]
            return result

        @self.server.tool()
        def list_releases(project_id: str, per_page: int | None = None) -> list[dict[str, Any]]:
            ours(project_id)
            return [{"tag_name": "infra-2026.09", "released_at": "2026-09-29T10:00:00Z"}]

        @self.server.tool()
        def get_repository_file(project_id: str, file_path: str, ref: str) -> dict[str, Any]:
            self.calls.append(("get_repository_file", {"path": file_path, "ref": ref}))
            ours(project_id)
            if file_path not in self.files:
                raise ToolError("404 File Not Found")
            return {"file_path": file_path, "commit_id": GL_SHA, "content": self.files[file_path]}


def json(value: Any) -> str:
    import json as j

    return j.dumps(value)


@pytest.fixture
def github() -> RepoGitHub:
    return RepoGitHub()


@pytest.fixture
def gitlab() -> FakeGitLab:
    return FakeGitLab()


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])


def config(**changes) -> Settings:
    values = {
        "github_mcp_url": "http://unused.invalid/github",
        "gitlab_mcp_url": f"{GITLAB}/api/v4/mcp",
    }
    return Settings(_env_file=None, **(values | changes))


async def connect(store, repos=(APP, WEB_REPO), projects=(INFRA,)) -> None:
    await store.save_settings(
        TeamSettings(
            team_id=TEAM.id,
            github=GitHubSettings(repos=[CodeRepo(path=r) for r in repos]),
            gitlab=GitLabSettings(projects=[CodeRepo(path=p) for p in projects]),
        )
    )


def toolbox(github: RepoGitHub, gitlab: FakeGitLab | None = None, repos=(APP, WEB_REPO)):
    readers = [GitHubReader(r, github.server) for r in repos]
    lab = [GitLabReader(INFRA, gitlab.server, web=GITLAB)] if gitlab else None
    return TeamToolbox(
        TEAM.id,
        ALEX.id,
        InMemoryStore(),
        members=[],
        memory=None,
        jira="off",
        github=readers,
        gitlab=lab,
    )


def question(text: str) -> Question:
    return Question(
        id="q-1",
        team_id=TEAM.id,
        text=text,
        asker_id=ALEX.id,
        asker_name=ALEX.name,
        visibility="public",
    )


# readers from settings


def test_each_connected_repository_gets_a_reader_at_its_ref():
    team = TeamSettings(
        team_id=TEAM.id,
        github=GitHubSettings(repos=[CodeRepo(path=APP, ref="release"), CodeRepo(path=WEB_REPO)]),
        gitlab=GitLabSettings(projects=[CodeRepo(path=INFRA, ref="main")]),
    )

    github = github_readers(config(), team)
    gitlab = gitlab_readers(config(), team)

    assert [(r.full_name, r.ref) for r in github] == [(APP, "release"), (WEB_REPO, None)]
    assert [(r.full_name, r.ref, r.web) for r in gitlab] == [(INFRA, "main", GITLAB)]


def test_without_gitlab_projects_gitlab_is_off_the_menu_and_without_a_url_unavailable():
    none = TeamSettings(team_id=TEAM.id)
    some = TeamSettings(team_id=TEAM.id, gitlab=GitLabSettings(projects=[CodeRepo(path=INFRA)]))

    assert gitlab_readers(config(), none) is None
    assert "GITLAB_MCP_URL" in gitlab_readers(config(gitlab_mcp_url=None), some)
    assert github_readers(config(github_repo="acme/solo"), none)[0].full_name == "acme/solo"


def test_the_gitlab_web_address_comes_from_its_mcp_endpoint():
    assert web_address("https://gitlab.example.com/api/v4/mcp") == "https://gitlab.example.com"
    assert web_address("http://localhost:8743/mcp") == "https://gitlab.com"


def test_a_repository_is_picked_by_its_path_or_unambiguous_name(github):
    readers = [GitHubReader(r, github.server) for r in (APP, WEB_REPO, "other/web")]

    assert [r.full_name for r in pick(readers, "DROPSUBS/web")] == [WEB_REPO]
    assert [r.full_name for r in pick(readers, "app")] == [APP]
    assert len(pick(readers, None)) == 3
    with pytest.raises(LookupError, match="dropsubs/app"):
        pick(readers, "web")  # two repositories are called web
    with pytest.raises(LookupError):
        pick(readers, "nowhere/else")


# the menu


async def test_the_menu_names_the_connected_repositories_and_offers_a_repo_argument(github, gitlab):
    specs = {s.name: s for s in toolbox(github, gitlab).specs()}
    without = {s.name for s in toolbox(github).specs()}

    assert f"Repositories: {APP}, {WEB_REPO}." in specs["github_search"].description
    assert f"Projects: {INFRA}." in specs["gitlab_read"].description
    assert "repo" in specs["github_code"].parameters["properties"]
    assert {"gitlab_search", "gitlab_read", "gitlab_releases", "gitlab_code"} <= set(specs)
    assert not {n for n in without if n.startswith("gitlab_")}


# GitHub across repositories


async def test_a_search_without_a_repo_covers_every_repository_and_names_each(github):
    result = await toolbox(github).call("github_search", {"query": "30%"})

    assert result.ok and result.error is None
    assert [f.source.label for f in result.content] == [f"{WEB_REPO}#41", f"{WEB_REPO}#7"]
    assert {c[1]["repo"] for c in github.calls if c[0] == "search_issues"} == {APP, WEB_REPO}


async def test_a_named_repository_is_the_only_one_read(github):
    result = await toolbox(github).call("github_search", {"query": "fee", "repo": "web"})

    assert [f.source.label for f in result.content] == [f"{WEB_REPO}#7"]
    assert {c[1]["repo"] for c in github.calls} == {WEB_REPO}


async def test_the_same_number_in_two_repositories_is_not_confused(github):
    both = await toolbox(github).call("github_read", {"number": 41})
    one = await toolbox(github).call("github_read", {"number": 7})

    assert both.ok and both.error is None
    labels = {f.source.label: f.text for f in both.content}
    assert set(labels) == {f"{APP}#41", f"{WEB_REPO}#41"}
    assert "Inbox sync" in labels[f"{APP}#41"] and "Pricing" in labels[f"{WEB_REPO}#41"]
    assert [f.source.label for f in one.content] == [f"{WEB_REPO}#7"]  # 404 in the app is fine


async def test_an_unknown_repository_is_a_tool_error_naming_the_connected_ones(github):
    result = await toolbox(github).call("github_read", {"number": 1, "repo": "acme/secret"})

    assert not result.ok
    assert APP in result.error and WEB_REPO in result.error
    assert github.calls == []


async def test_a_failing_repository_is_reported_while_the_others_answer(github):
    github.broken.add(APP)

    result = await toolbox(github).call("github_search", {"query": "fee"})

    assert result.ok
    assert [f.source.label for f in result.content] == [f"{WEB_REPO}#7"]
    assert APP in result.error and "502" in result.error


async def test_releases_of_every_repository_are_labelled_with_it(github):
    result = await toolbox(github).call("github_releases", {})

    assert [f.source.label for f in result.content] == [f"{APP}@v0.9.3"]
    assert {c[1]["repo"] for c in github.calls if c[0] == "list_releases"} == {APP, WEB_REPO}


async def test_code_search_spans_repositories_and_names_each_snippet(github):
    result = await toolbox(github).call("github_code", {"query": "fee"})

    assert result.ok
    labels = [f.source.label for f in result.content]
    assert labels == [f"{APP}:api/billing/fees.py L1-L2", f"{WEB_REPO}:lib/pricing.ts L1-L2"]
    snippet = result.content[1].snippet
    assert snippet.repo == WEB_REPO
    assert snippet.github_url == f"https://github.com/{WEB_REPO}/blob/{SHA}/lib/pricing.ts#L1-L2"
    assert snippet.caption.startswith(f"{WEB_REPO}: lib/pricing.ts")


async def test_code_evidence_takes_files_in_turn_from_each_repository(github):
    github.repos[APP]["files"] |= {"api/a.py": "fee\n", "api/b.py": "fee\n"}
    readers = [GitHubReader(r, github.server) for r in (APP, WEB_REPO)]

    snippets = await code_evidence_across(readers, "fee", max_files=2)

    assert [s.repo for s in snippets] == [APP, WEB_REPO]


# GitLab


async def test_gitlab_search_cites_merge_requests_with_a_bang_and_drops_other_projects(
    github, gitlab
):
    mrs = await toolbox(github, gitlab).call("gitlab_search", {"query": "status", "kind": "mr"})
    issues = await toolbox(github, gitlab).call("gitlab_search", {"query": "load test"})

    (mr,) = mrs.content
    assert mr.source == Source(
        kind="gitlab_mr", label=f"{INFRA}!4", url=f"{GITLAB}/{INFRA}/-/merge_requests/4"
    )
    assert "(merged 2026-09-28)" in mr.text
    assert [f.source.label for f in issues.content] == [f"{INFRA}#5"]  # not other/secret#9
    assert issues.content[0].source.kind == "gitlab_issue"


async def test_gitlab_reads_an_issue_in_the_work_item_shape(github, gitlab):
    result = await toolbox(github, gitlab).call("gitlab_read", {"number": 5, "kind": "issue"})

    (issue,) = result.content
    assert issue.source.label == f"{INFRA}#5"
    assert "Load test before the waitlist opens (opened)" in issue.text


async def test_a_merge_request_reads_with_the_files_it_changes(github, gitlab):
    result = await toolbox(github, gitlab).call("gitlab_read", {"number": 4, "kind": "mr"})

    mr, diff = result.content
    assert mr.source.label == diff.source.label == f"{INFRA}!4"
    assert "changes monitoring/status.yml (added)" in diff.text
    assert "+interval_seconds: 60" in diff.text
    assert ("get_merge_request", {"iid": 4, "include": ["diffs"]}) in gitlab.calls


async def test_a_failed_diff_keeps_the_merge_request_and_says_what_failed(github, gitlab):
    gitlab.diff_fails = True

    result = await toolbox(github, gitlab).call("gitlab_read", {"number": 4, "kind": "mr"})

    assert result.ok
    assert [f.source.label for f in result.content] == [f"{INFRA}!4"]
    assert "diffs are too large" in result.error


async def test_gitlab_code_is_cited_with_a_gitlab_permalink(github, gitlab):
    result = await toolbox(github, gitlab).call("gitlab_code", {"query": "interval_seconds"})

    (code,) = result.content
    assert code.source.kind == "gitlab_code"
    assert code.source.label == f"{INFRA}:monitoring/status.yml L1-L2"
    assert code.snippet.github_url == (
        f"{GITLAB}/{INFRA}/-/blob/{GL_SHA}/monitoring/status.yml#L1-2"
    )
    assert ("get_repository_file", {"path": "monitoring/status.yml", "ref": "HEAD"}) in gitlab.calls


async def test_gitlab_releases_are_labelled_with_the_project(github, gitlab):
    result = await toolbox(github, gitlab).call("gitlab_releases", {})

    (release,) = result.content
    assert release.source.kind == "gitlab_release"
    assert release.source.label == f"{INFRA}@infra-2026.09"
    assert release.source.url == f"{GITLAB}/{INFRA}/-/releases/infra-2026.09"


async def test_a_reader_never_reads_outside_its_project(gitlab):
    reader = GitLabReader(INFRA, gitlab.server, web=GITLAB)

    with pytest.raises(GitLabError):
        await reader.read(99, "mr")
    with pytest.raises(ValueError):
        GitLabReader("../etc", gitlab.server)


async def test_gitlab_tools_say_so_when_gitlab_is_not_configured(github):
    box = TeamToolbox(
        TEAM.id,
        ALEX.id,
        InMemoryStore(),
        members=[],
        memory=None,
        jira="off",
        github=[GitHubReader(APP, github.server)],
        gitlab="GitLab is not configured: set GITLAB_MCP_URL",
    )

    result = await box.call("gitlab_search", {"query": "status"})

    assert not result.ok and "GITLAB_MCP_URL" in result.error
    assert "gitlab_search" in {s.name for s in box.specs()}  # on the menu, marked unavailable
    assert "GITLAB_MCP_URL" in box.unavailable("gitlab_search")


# Ask, end to end with the fakes


async def test_ask_answers_from_the_repository_the_question_names(store, github, gitlab):
    await connect(store)
    llm = scripted(
        PlannedCall(tool="github_search", query="fee", repo=WEB_REPO),
        PlannedCall(tool="gitlab_read", number=4, kind="mr"),
        answer=citing(f"{WEB_REPO}#7", f"{INFRA}!4", text="The landing page fee is fixed."),
    )
    orchestrator = ToolOrchestrator(
        llm, store, settings=config(), github_target=github.server, gitlab_target=gitlab.server
    )

    answer = await orchestrator.ask(question("Is the fee on the website fixed?"))

    assert [s.label for s in answer.sources] == [f"{WEB_REPO}#7", f"{INFRA}!4"]
    assert [s.kind for s in answer.sources] == ["github_issue", "gitlab_mr"]
    plan, reply = llm.calls
    assert f"Repositories: {APP}, {WEB_REPO}." in plan.prompt
    assert f"Projects: {INFRA}." in plan.prompt
    assert "repo" in AskPlan.model_json_schema()["$defs"]["PlannedCall"]["properties"]
    assert all(APP not in line for line in evidence(reply.prompt).values())


# fact-checks


async def test_fact_check_code_evidence_covers_every_repository_and_names_it(store, github, gitlab):
    await connect(store)
    meeting = await store.create_meeting(TEAM.id, "Standup", ALEX.id)
    await store.add_participant(meeting.id, SARAH.id)
    await store.add_segments(
        meeting.id,
        [
            TranscriptSegment(
                seg_id="s1",
                meeting_id=meeting.id,
                speaker_id=SARAH.id,
                speaker_name=SARAH.name,
                text="The status page checks every 5 minutes, the timeout is 300 seconds.",
                t_start=10,
                t_end=14,
                is_final=True,
            )
        ],
    )
    looked_up: list[str] = []

    async def code(reader, query: str):
        looked_up.append(reader.full_name)
        from brain.agent.code import code_evidence

        return await code_evidence(reader, query)

    def verdicts(prompt: str) -> FactCheckVerdicts:
        ids = [i for i, line in evidence(prompt).items() if INFRA in line]
        return FactCheckVerdicts(
            checks=[
                ClaimVerdict(
                    claim="c1",
                    verdict="contradicted",
                    confidence=0.9,
                    severity="low",
                    evidence_ids=ids,
                    finding="It checks every 60 seconds.",
                )
            ]
        )

    llm = MockLLM(
        structured={
            FactCheckPlan: FactCheckPlan(
                claims=[ClaimPlan(claim="c1", code_query="interval_seconds")]
            ),
            FactCheckVerdicts: verdicts,
        }
    )
    checker = FactChecker(
        llm,
        store,
        settings=config(),
        github_target=github.server,
        gitlab_target=gitlab.server,
        code=code,
    )

    response = await checker.tick(meeting, 60)

    assert sorted(looked_up) == sorted([APP, WEB_REPO, INFRA])
    (check,) = response.checks
    assert [(s.kind, s.label) for s in check.sources] == [
        ("gitlab_code", f"{INFRA}:monitoring/status.yml:1-2")
    ]
    (plan_prompt, _) = [c.prompt for c in llm.calls]
    assert f"Repositories: {APP}, {WEB_REPO}, {INFRA}." in plan_prompt


# keyterms


async def test_keyterms_name_every_repository_and_project(store):
    await connect(store)
    meeting = await store.create_meeting(TEAM.id, "Standup", ALEX.id)

    terms = await meeting_keyterms(store, Settings(_env_file=None), meeting)

    assert {"app", "web", "infra"} <= set(terms)
