"""The read-only tools the agent may call for one team. Each result is a list of findings, every
one with the Source a person can check: a meeting moment, a Jira key, or an item of one of the
team's GitHub repositories or GitLab projects, named with its repository."""

from collections.abc import Awaitable, Callable, Coroutine, Sequence
from datetime import UTC, date, tzinfo
from typing import Any, Literal

import anyio
import httpx
from pydantic import BaseModel

from brain.config import Settings
from brain.github import GitHubItem, GitHubReader, GitHubRelease
from brain.github_account import Unusable, repo_target
from brain.gitlab import GitLabDiff, GitLabItem, GitLabReader, web_address
from brain.integrations import McpEndpoint, ToolRefused
from brain.jira import JiraConfig, JiraIssue, JiraReader, root_cause
from brain.jira_rest import JiraRestReader, account_access, reads_with_account
from brain.memory import Chunk, MeetingMemory
from brain.memory.chunking import Turn, people_turns, turn_text
from brain.report.decisions import terms
from brain.report.extraction import clock
from brain.store import JiraAccount, NotFound, Store
from brain.zones import local_date
from brain.zones import today as team_today
from contracts import (
    CodeRepo,
    CodeSnippet,
    Meeting,
    Person,
    Source,
    TeamSettings,
    TranscriptSegment,
)
from contracts.agent import SourceKind

from .code import CodeReader, code_evidence_across
from .tools import ToolResult, ToolSpec

TaskStatus = Literal["open", "overdue", "all"]

MEMORY_HITS = 6
MEMORY_FINDINGS = 24
"""Most findings one meeting search returns: a transcript hit is a window of several turns."""
LIST_LIMIT = 10
TASK_LIMIT = 15
DEFAULT_TIMEOUT = 10.0
SEARCH_LIMIT = 8  # issues or pull requests from one search, across the repositories searched
PER_REPO_SEARCH = 6
MANY_REPO_RELEASES = 2  # releases per repository when several are read at once


class Finding(BaseModel):
    text: str
    source: Source
    when: date | None = None
    snippet: CodeSnippet | None = None  # code: the lines `text` shows, copied from the file


QUERY = {"query": {"type": "string", "description": "Short search text: the topic's key words."}}
REPO = {
    "repo": {
        "type": "string",
        "description": "One of the listed repositories, when the question names one; leave it "
        "out to read all of them.",
    }
}

SPECS: list[tuple[ToolSpec, str]] = [
    (
        ToolSpec(
            name="search_meetings",
            description="Search the team's past meetings: transcripts, summaries, decisions and "
            "tasks, each with its meeting and moment.",
            parameters={"type": "object", "properties": QUERY, "required": ["query"]},
        ),
        "Meeting memory",
    ),
    (
        ToolSpec(
            name="decisions",
            description="The team's recorded decisions, newest first, optionally only those "
            "about the query.",
            parameters={"type": "object", "properties": QUERY},
        ),
        "Decisions",
    ),
    (
        ToolSpec(
            name="tasks",
            description="The team's tasks with owner, due date and status. owner_id is a team "
            'member id, or "me" for the asker; leave it out for everyone. status is open '
            "(default), overdue or all.",
            parameters={
                "type": "object",
                "properties": {
                    "owner_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "overdue", "all"]},
                },
            },
        ),
        "Tasks",
    ),
    (
        ToolSpec(
            name="recent_meetings",
            description="The team's latest meetings with their dates and summaries.",
            parameters={"type": "object", "properties": {}},
        ),
        "Meetings",
    ),
    (
        ToolSpec(
            name="jira_search",
            description="Search the team's Jira project; returns live status, assignee and "
            "priority.",
            parameters={"type": "object", "properties": QUERY, "required": ["query"]},
        ),
        "Jira",
    ),
    (
        ToolSpec(
            name="jira_issue",
            description="Read one Jira issue of the team's project by key, e.g. DS-104.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        ),
        "Jira",
    ),
    (
        ToolSpec(
            name="github_search",
            description="Search the team's GitHub repositories' issues (kind issue) or pull "
            "requests (kind pr).",
            parameters={
                "type": "object",
                "properties": {
                    **QUERY,
                    "kind": {"type": "string", "enum": ["issue", "pr"]},
                    **REPO,
                },
                "required": ["query"],
            },
        ),
        "GitHub",
    ),
    (
        ToolSpec(
            name="github_read",
            description="Read one issue (kind issue) or pull request (kind pr) of the team's "
            "GitHub repositories by number.",
            parameters={
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["issue", "pr"]},
                    **REPO,
                },
                "required": ["number"],
            },
        ),
        "GitHub",
    ),
    (
        ToolSpec(
            name="github_releases",
            description="The team's GitHub repositories' latest releases, newest first, with "
            "their publish dates. A pull request merged after the latest release is merged but "
            "not released yet.",
            parameters={"type": "object", "properties": {**REPO}},
        ),
        "GitHub",
    ),
    (
        ToolSpec(
            name="github_code",
            description="Search the code of the team's GitHub repositories; returns the matching "
            "lines of the top files, numbered, with links.",
            parameters={"type": "object", "properties": {**QUERY, **REPO}, "required": ["query"]},
        ),
        "GitHub code",
    ),
    (
        ToolSpec(
            name="gitlab_search",
            description="Search the team's GitLab projects' issues (kind issue) or merge "
            "requests (kind mr).",
            parameters={
                "type": "object",
                "properties": {
                    **QUERY,
                    "kind": {"type": "string", "enum": ["issue", "mr"]},
                    **REPO,
                },
                "required": ["query"],
            },
        ),
        "GitLab",
    ),
    (
        ToolSpec(
            name="gitlab_read",
            description="Read one issue (kind issue) or merge request (kind mr, with the files "
            "it changes and their diffs) of the team's GitLab projects by number.",
            parameters={
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["issue", "mr"]},
                    **REPO,
                },
                "required": ["number"],
            },
        ),
        "GitLab",
    ),
    (
        ToolSpec(
            name="gitlab_releases",
            description="The team's GitLab projects' latest releases, newest first.",
            parameters={"type": "object", "properties": {**REPO}},
        ),
        "GitLab",
    ),
    (
        ToolSpec(
            name="gitlab_code",
            description="Search the code of the team's GitLab projects; returns the matching "
            "lines of the top files, numbered, with links.",
            parameters={"type": "object", "properties": {**QUERY, **REPO}, "required": ["query"]},
        ),
        "GitLab code",
    ),
]

RELEASE_LIMIT = 5

SOURCE_NAMES = {spec.name: name for spec, name in SPECS}


JIRA_CONNECT_AGAIN = (
    "Jira is not reachable: the token the project was connected with can't be read on this "
    "server: an admin must connect Jira again"
)


def jira_reader(
    settings: Settings,
    team: TeamSettings,
    target: Any = None,
    account: JiraAccount | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> JiraReader | str:
    """A reader for the team's Jira project, or why there is none. The project the team's
    account was connected with is read on its own site with that account (unless it is one of
    the demo world's, MOCK_JIRA_PROJECTS); any other through JIRA_MCP_URL. A team without its
    own project uses the deployment's JIRA_PROJECT_KEY, as the push flow does (one team per
    deployment). `target` (tests) stands in for the MCP server."""
    if target is None and reads_with_account(settings, team.jira.project, team.jira.site, account):
        assert account is not None
        access = account_access(settings, account)
        if access is None:
            return JIRA_CONNECT_AGAIN
        return JiraRestReader(access, transport=transport)
    cloud_id = settings.jira_cloud_id or settings.jira_base_url
    project = team.jira.project or settings.jira_project_key
    missing = [
        name
        for name, value in {
            "JIRA_MCP_URL": settings.jira_mcp_url,
            "JIRA_CLOUD_ID (or JIRA_BASE_URL)": cloud_id,
        }.items()
        if not value
    ]
    if missing:
        return f"Jira is not configured: set {', '.join(missing)}"
    if not project:
        return "Jira is not configured: no Jira project is set in workspace settings"
    config = JiraConfig(
        mcp_url=settings.jira_mcp_url,
        cloud_id=cloud_id,
        project_key=project,
        base_url=settings.jira_base_url or None,
    )
    return JiraReader(config, target=target)


def code_repos(team: TeamSettings, settings: Settings) -> list[CodeRepo]:
    """The team's GitHub repositories; a team without any uses the deployment's GITHUB_REPO,
    as the push flow does with Jira (one team per deployment)."""
    if team.github.repos:
        return team.github.repos
    return [CodeRepo(path=settings.github_repo)] if settings.github_repo else []


def github_readers(
    settings: Settings,
    team: TeamSettings,
    target: Any = None,
    endpoints: dict[str, McpEndpoint | str] | None = None,
) -> list[GitHubReader] | str:
    """A reader for each of the team's repositories at its ref (default branch when unset), or
    why there is none. `endpoints` are the repositories connected with a token
    (github_account.github_endpoints), read through GitHub's hosted server with it; the demo
    world's (MOCK_GITHUB_OWNERS) and any without one are read from GITHUB_MCP_URL with no
    credentials. `target` (tests) stands in for whichever server a repository is read from."""
    repos = code_repos(team, settings)
    if not repos and not settings.github_mcp_url:
        return (
            "GitHub is not configured: connect a repository with its token in Settings, or set "
            "GITHUB_MCP_URL"
        )
    if not repos:
        return "GitHub is not configured: no repository is set in workspace settings or GITHUB_REPO"
    readers, problems = [], []
    for repo in repos:
        try:
            server = repo_target(settings, repo.path, endpoints or {})
            readers.append(GitHubReader(repo.path, target or server, ref=repo.ref))
        except Unusable as e:
            problems.append(str(e))
        except ValueError as e:
            problems.append(f"GitHub is not configured: {e}")
    return readers or problems[0]


def gitlab_readers(
    settings: Settings, team: TeamSettings, target: Any = None
) -> list[GitLabReader] | str | None:
    """A reader for each of the team's GitLab projects, or why there is none; None when the
    team connects no GitLab project, so GitLab stays off the agent's menu."""
    if not team.gitlab.projects:
        return None
    if not settings.gitlab_mcp_url:
        return "GitLab is not configured: set GITLAB_MCP_URL"
    web = web_address(settings.gitlab_mcp_url)
    readers, problems = [], []
    for project in team.gitlab.projects:
        try:
            readers.append(
                GitLabReader(
                    project.path, target or settings.gitlab_mcp_url, ref=project.ref, web=web
                )
            )
        except ValueError as e:
            problems.append(str(e))
    return readers or f"GitLab is not configured: {problems[0]}"


class LookupFailed(Exception):
    """Some repositories answered and some failed: the findings, and what failed."""

    def __init__(self, findings: list["Finding"], errors: list[str]):
        super().__init__("; ".join(errors))
        self.findings = findings
        self.errors = errors


def pick[R: (GitHubReader, GitLabReader)](readers: Sequence[R], repo: str | None) -> list[R]:
    """The readers a call is about: all of them, or the one `repo` names, by its full path or,
    when that is unambiguous, its last part ("website" for dropsubs/website)."""
    if not repo or not repo.strip():
        return list(readers)
    wanted = repo.strip().strip("/").casefold()
    exact = [r for r in readers if r.full_name.casefold() == wanted]
    if exact:
        return exact
    tail = wanted.rsplit("/", 1)[-1]
    named = [r for r in readers if r.full_name.rsplit("/", 1)[-1].casefold() == tail]
    if len(named) == 1:
        return named
    names = ", ".join(r.full_name for r in readers)
    raise LookupError(f"{repo} is not one of the connected repositories ({names})")


def interleave(groups: Sequence[Sequence["Finding"]], limit: int) -> list["Finding"]:
    """Findings taken in turn from each group, `limit` in all."""
    found: list[Finding] = []
    for depth in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if depth < len(group) and len(found) < limit:
                found.append(group[depth])
    return found


class TeamToolbox:
    """Every tool reads only `team_id`'s data. Unconfigured tools stay on the menu, marked, so a
    question that needs them can say they are unavailable; GitLab's are on it only when the team
    connects a GitLab project. A repository tool reads every connected repository unless the
    call names one. Dates are the team's, in `zone`."""

    def __init__(
        self,
        team_id: str,
        asker_id: str,
        store: Store,
        *,
        members: list[Person],
        memory: MeetingMemory | None,
        jira: JiraReader | str,
        github: GitHubReader | Sequence[GitHubReader] | str,
        gitlab: Sequence[GitLabReader] | str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        today: date | None = None,
        zone: tzinfo = UTC,
    ):
        self.team_id = team_id
        self.asker_id = asker_id
        self.store = store
        self.members = {p.id: p for p in members}
        self.memory = memory
        self.jira = jira
        self.github: list[GitHubReader] | str = (
            [github]
            if isinstance(github, GitHubReader)
            else github
            if isinstance(github, str)
            else list(github)
        )
        self.gitlab: list[GitLabReader] | str | None = (
            gitlab if gitlab is None or isinstance(gitlab, str) else list(gitlab)
        )
        self.timeout = timeout
        self.zone = zone
        self.today = today or team_today(zone)
        self._meetings: dict[str, Meeting | None] = {}
        self._transcripts: dict[str, list[TranscriptSegment]] = {}
        self._tools: dict[str, Callable[..., Coroutine[Any, Any, list[Finding]]]] = {
            "search_meetings": self.search_meetings,
            "decisions": self.decisions,
            "tasks": self.tasks,
            "recent_meetings": self.recent_meetings,
            "jira_search": self.jira_search,
            "jira_issue": self.jira_issue,
            "github_search": self.github_search,
            "github_read": self.github_read,
            "github_releases": self.github_releases,
            "github_code": self.github_code,
            "gitlab_search": self.gitlab_search,
            "gitlab_read": self.gitlab_read,
            "gitlab_releases": self.gitlab_releases,
            "gitlab_code": self.gitlab_code,
        }

    def specs(self) -> list[ToolSpec]:
        """The menu. Repository tools name the connected repositories."""
        specs = []
        for spec, _ in SPECS:
            readers = self.readers_for(spec.name)
            if spec.name.startswith("gitlab_") and self.gitlab is None:
                continue
            if isinstance(readers, list):
                names = ", ".join(r.full_name for r in readers)
                noun = "Projects" if spec.name.startswith("gitlab_") else "Repositories"
                spec = spec.model_copy(
                    update={"description": f"{spec.description} {noun}: {names}."}
                )
            specs.append(spec)
        return specs

    def readers_for(self, name: str) -> list[GitHubReader] | list[GitLabReader] | str | None:
        if name.startswith("github_"):
            return self.github
        if name.startswith("gitlab_"):
            return self.gitlab
        return None

    def code_readers(self) -> list[CodeReader]:
        """Every connected repository's code, GitHub's then GitLab's."""
        readers: list[CodeReader] = []
        for found in (self.github, self.gitlab):
            if isinstance(found, list):
                readers += found
        return readers

    def unavailable(self, name: str) -> str | None:
        """Why the tool cannot run right now, or None when it can."""
        if name == "search_meetings" and self.memory is None:
            return "Meeting memory is unavailable: no database or embedding model is configured"
        if name.startswith("jira_") and isinstance(self.jira, str):
            return self.jira
        if name.startswith("github_") and isinstance(self.github, str):
            return self.github
        if name.startswith("gitlab_") and not isinstance(self.gitlab, list):
            return self.gitlab or "No GitLab project is connected in workspace settings"
        return None

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """content is a list of Findings. Fails into ToolResult.error, never raises, except for
        a tool that is not on the menu."""
        if name not in self._tools:
            raise ToolRefused(f"{name} is not one of the agent's read tools")
        if reason := self.unavailable(name):
            return ToolResult(name=name, ok=False, error=reason)
        source = SOURCE_NAMES[name]
        try:
            with anyio.fail_after(self.timeout):
                findings = await self._tools[name](**arguments)
        except TimeoutError:
            error = f"{source} timed out after {self.timeout:g}s"
            return ToolResult(name=name, ok=False, error=error)
        except LookupFailed as p:
            error = f"{source} failed: {'; '.join(p.errors)}"
            return ToolResult(name=name, ok=True, content=p.findings, error=error)
        except ToolRefused:
            raise
        except Exception as e:
            return ToolResult(name=name, ok=False, error=f"{source} failed: {root_cause(e)}")
        return ToolResult(name=name, ok=True, content=findings)

    # the tools

    async def search_meetings(self, query: str | None = None, **_: Any) -> list[Finding]:
        assert self.memory is not None
        found = []
        for hit in await self.memory.search(self.team_id, query or "", k=MEMORY_HITS):
            chunk = hit.chunk
            meeting = await self.meeting(chunk.meeting_id)
            if chunk.team_id != self.team_id or meeting is None:
                continue
            if turns := await self.turns_in(chunk):
                found += [self.at(meeting, turn[0].t_start, turn_text(turn)) for turn in turns]
            else:
                found.append(self.at(meeting, chunk.t_start, chunk.text))
        return found[:MEMORY_FINDINGS]

    async def turns_in(self, chunk: Chunk) -> list[Turn]:
        """A transcript chunk's turns from the stored transcript, so each is cited at its own
        moment rather than the window's start. Empty for other chunks or a missing transcript."""
        if chunk.kind != "transcript" or chunk.t_start is None or chunk.t_end is None:
            return []
        if chunk.meeting_id not in self._transcripts:
            self._transcripts[chunk.meeting_id] = await self.store.transcript(chunk.meeting_id)
        inside = [
            s
            for s in self._transcripts[chunk.meeting_id]
            if chunk.t_start <= s.t_start and s.t_end <= chunk.t_end
        ]
        return people_turns(chunk.meeting_id, inside)

    async def decisions(self, query: str | None = None, **_: Any) -> list[Finding]:
        decisions = await self.store.decisions(self.team_id)
        if words := terms(query or ""):
            scored = [(len(words & terms(d.text)), i, d) for i, d in enumerate(decisions)]
            decisions = [d for score, _, d in sorted(scored, key=lambda s: (-s[0], s[1])) if score]
        found = []
        for d in decisions[:LIST_LIMIT]:
            meeting = await self.meeting(d.meeting_id)
            if meeting is None:
                continue
            status = " (superseded by a later decision)" if d.status == "superseded" else ""
            by = self.name(d.made_by)
            text = f"Decision{status}: {d.text}." + (f" Made by {by}." if by else "")
            if d.quote:
                text += f' They said: "{d.quote}"'
            found.append(self.at(meeting, d.t, text))
        return found

    async def tasks(
        self, owner_id: str | None = None, status: TaskStatus | None = None, **_: Any
    ) -> list[Finding]:
        if owner_id and owner_id.strip().lower() in ("me", "asker", "self"):
            owner_id = self.asker_id
        if owner_id and owner_id not in self.members:
            raise LookupError(f"{owner_id} is not a member of the team")
        status = status or "open"
        tasks = [t for t in await self.store.tasks(self.team_id, owner_id or None) if t.include]
        if status != "all":
            tasks = [t for t in tasks if t.jira_status != "done"]
        if status == "overdue":
            tasks = [t for t in tasks if t.due is not None and t.due < self.today]
        tasks.sort(key=lambda t: (t.due is None, t.due or self.today))
        found = []
        for t in tasks[:TASK_LIMIT]:
            meeting = await self.meeting(t.meeting_id)
            if meeting is None:
                continue
            owner = self.name(t.owner_id) or "unassigned"
            due = f"due {t.due.isoformat()}" if t.due else "no due date"
            text = f"Task: {t.title} ({owner}, {due}, {t.jira_status.replace('_', ' ')}"
            text += f", Jira {t.key})" if t.key else ")"
            if t.description:
                text += f". {t.description}"
            found.append(self.at(meeting, t.t, text))
        return found

    async def recent_meetings(self, **_: Any) -> list[Finding]:
        found = []
        for meeting in (await self.store.meetings(self.team_id))[:LIST_LIMIT]:
            self._meetings[meeting.id] = meeting
            when = day(meeting, self.zone)
            text = f'Meeting "{meeting.title}" ({meeting.status.replace("_", " ")})'
            try:
                summary = (await self.store.report(meeting.id)).summary.strip()
            except NotFound:
                summary = ""
            if summary:
                text += f". Summary: {summary}"
            source = Source(kind="meeting", label=meeting.title, meeting_id=meeting.id)
            found.append(Finding(text=text, source=source, when=when))
        return found

    async def jira_search(self, query: str | None = None, **_: Any) -> list[Finding]:
        assert isinstance(self.jira, JiraReader)
        return [jira_finding(issue) for issue in await self.jira.search(query or "")]

    async def jira_issue(self, key: str | None = None, **_: Any) -> list[Finding]:
        assert isinstance(self.jira, JiraReader)
        return [jira_finding(await self.jira.get(key))] if key else []

    async def github_search(
        self, query: str | None = None, kind: str | None = None, repo: str | None = None, **_: Any
    ) -> list[Finding]:
        assert isinstance(self.github, list)
        if not (query or "").strip():
            return []
        wanted = "pr" if kind == "pr" else "issue"

        async def search(reader: GitHubReader) -> list[Finding]:
            items = await reader.search(query or "", wanted, PER_REPO_SEARCH)
            return [github_finding(reader.full_name, item, self.zone) for item in items]

        return await across(pick(self.github, repo), search, SEARCH_LIMIT)

    async def github_read(
        self, number: int | None = None, kind: str | None = None, repo: str | None = None, **_: Any
    ) -> list[Finding]:
        assert isinstance(self.github, list)
        if number is None:
            return []
        wanted = "pr" if kind == "pr" else "issue"

        async def read(reader: GitHubReader) -> list[Finding]:
            item = await reader.read(number, wanted)
            return [github_finding(reader.full_name, item, self.zone)]

        return await across(pick(self.github, repo), read, LIST_LIMIT, missing_is_fine=True)

    async def github_releases(self, repo: str | None = None, **_: Any) -> list[Finding]:
        assert isinstance(self.github, list)
        readers = pick(self.github, repo)
        each = RELEASE_LIMIT if len(readers) == 1 else MANY_REPO_RELEASES

        async def releases(reader: GitHubReader) -> list[Finding]:
            found = await reader.releases(each)
            return [
                release_finding(reader.full_name, r, self.zone, latest=i == 0)
                for i, r in enumerate(found)
            ]

        return await across(readers, releases, LIST_LIMIT)

    async def github_code(
        self, query: str | None = None, repo: str | None = None, **_: Any
    ) -> list[Finding]:
        assert isinstance(self.github, list)
        if not (query or "").strip():
            return []
        snippets = await code_evidence_across(pick(self.github, repo), query or "")
        return [code_finding(s, "github_code") for s in snippets]

    async def gitlab_search(
        self, query: str | None = None, kind: str | None = None, repo: str | None = None, **_: Any
    ) -> list[Finding]:
        assert isinstance(self.gitlab, list)
        if not (query or "").strip():
            return []
        wanted = "mr" if kind in ("mr", "pr") else "issue"

        async def search(reader: GitLabReader) -> list[Finding]:
            items = await reader.search(query or "", wanted, PER_REPO_SEARCH)
            return [gitlab_finding(reader, item, self.zone) for item in items]

        return await across(pick(self.gitlab, repo), search, SEARCH_LIMIT)

    async def gitlab_read(
        self, number: int | None = None, kind: str | None = None, repo: str | None = None, **_: Any
    ) -> list[Finding]:
        assert isinstance(self.gitlab, list)
        if number is None:
            return []
        wanted = "mr" if kind in ("mr", "pr") else "issue"

        async def read(reader: GitLabReader) -> list[Finding]:
            item = await reader.read(number, wanted)
            found = [gitlab_finding(reader, item, self.zone)]
            if wanted == "mr":
                try:
                    diffs = await reader.diff(number)
                except Exception as e:  # the merge request itself was read
                    error = f"{reader.label('mr', number)} diff: {root_cause(e)}"
                    raise LookupFailed(found, [error]) from e
                found += [diff_finding(found[0].source, d) for d in diffs]
            return found

        return await across(pick(self.gitlab, repo), read, LIST_LIMIT, missing_is_fine=True)

    async def gitlab_releases(self, repo: str | None = None, **_: Any) -> list[Finding]:
        assert isinstance(self.gitlab, list)
        readers = pick(self.gitlab, repo)
        each = RELEASE_LIMIT if len(readers) == 1 else MANY_REPO_RELEASES

        async def releases(reader: GitLabReader) -> list[Finding]:
            found = await reader.releases(each)
            return [
                release_finding(
                    reader.full_name, r, self.zone, latest=i == 0, kind="gitlab_release"
                )
                for i, r in enumerate(found)
            ]

        return await across(readers, releases, LIST_LIMIT)

    async def gitlab_code(
        self, query: str | None = None, repo: str | None = None, **_: Any
    ) -> list[Finding]:
        assert isinstance(self.gitlab, list)
        if not (query or "").strip():
            return []
        snippets = await code_evidence_across(pick(self.gitlab, repo), query or "")
        return [code_finding(s, "gitlab_code") for s in snippets]

    # helpers

    async def meeting(self, meeting_id: str) -> Meeting | None:
        """The meeting if it is this team's; anything else is as good as missing."""
        if meeting_id not in self._meetings:
            try:
                meeting = await self.store.meeting(meeting_id)
            except NotFound:
                meeting = None
            ours = meeting is not None and meeting.team_id == self.team_id
            self._meetings[meeting_id] = meeting if ours else None
        return self._meetings[meeting_id]

    def at(self, meeting: Meeting, t: float | None, text: str) -> Finding:
        return Finding(text=text, source=meeting_source(meeting, t), when=day(meeting, self.zone))

    def name(self, person_id: str | None) -> str | None:
        person = self.members.get(person_id or "")
        return person.name if person else None


async def across[R: (GitHubReader, GitLabReader)](
    readers: Sequence[R],
    read: Callable[[R], Awaitable[list[Finding]]],
    limit: int,
    *,
    missing_is_fine: bool = False,
) -> list[Finding]:
    """`read` for each repository at the same time; their findings in turn, `limit` in all. When
    every repository fails, the first failure is raised; when some do, LookupFailed carries what was
    found and what failed, unless `missing_is_fine` (a number read across repositories is
    expected to be missing from most of them) and something was found."""
    results: list[list[Finding] | Exception] = [[] for _ in readers]

    async def one(i: int, reader: R) -> None:
        try:
            results[i] = await read(reader)
        except ToolRefused:
            raise
        except LookupFailed as p:
            results[i] = p
        except Exception as e:
            results[i] = e

    async with anyio.create_task_group() as group:
        for i, reader in enumerate(readers):
            group.start_soon(one, i, reader)

    groups = [r.findings if isinstance(r, LookupFailed) else r for r in results]
    groups = [g for g in groups if isinstance(g, list)]
    errors: list[str] = []
    for reader, result in zip(readers, results, strict=True):
        if isinstance(result, LookupFailed):
            errors += result.errors
        elif isinstance(result, Exception):
            errors.append(f"{reader.full_name}: {root_cause(result)}")
    failed = [r for r in results if isinstance(r, Exception) and not isinstance(r, LookupFailed)]
    if readers and len(failed) == len(readers):
        if len(readers) == 1:
            raise failed[0]
        raise RuntimeError("; ".join(errors))
    found = interleave(groups, limit)
    if errors and not (
        missing_is_fine and found and not any(isinstance(r, LookupFailed) for r in results)
    ):
        raise LookupFailed(found, errors)
    return found


def meeting_source(meeting: Meeting, t: float | None) -> Source:
    label = meeting.title if t is None else f"{meeting.title} {clock(t)}"
    return Source(kind="meeting", label=label, meeting_id=meeting.id, t=t)


def day(meeting: Meeting, zone: tzinfo = UTC) -> date | None:
    """The meeting's day in the team's zone."""
    return local_date(meeting.started_at or meeting.scheduled_start, zone)


def jira_finding(issue: JiraIssue) -> Finding:
    details = [
        f"status {issue.status or 'unknown'}",
        f"assignee {issue.assignee or 'unassigned'}",
    ]
    if issue.priority:
        details.append(f"priority {issue.priority}")
    text = f"Jira {issue.key}: {issue.summary} ({', '.join(details)})"
    if issue.description:
        text += f". {issue.description}"
    return Finding(text=text, source=Source(kind="jira_issue", label=issue.key, url=issue.url))


def github_finding(repo: str, item: GitHubItem, zone: tzinfo = UTC) -> Finding:
    label = f"{repo}#{item.number}"
    state = item.state or "unknown"
    merged_on = local_date(item.merged_at, zone)
    if item.merged:
        state = "merged"
        if merged_on:
            state += f" {merged_on.isoformat()}"
    noun = "Pull request" if item.kind == "pr" else "Issue"
    text = f"{noun} {label}: {item.title} ({state})"
    if item.checks:
        text += f"; checks: {item.checks}"
    if item.reviews:
        text += f"; reviews: {item.reviews}"
    if item.body:
        text += f". {item.body}"
    kind = "github_pr" if item.kind == "pr" else "github_issue"
    return Finding(text=text, source=Source(kind=kind, label=label, url=item.url), when=merged_on)


def gitlab_finding(reader: GitLabReader, item: GitLabItem, zone: tzinfo = UTC) -> Finding:
    label = reader.label(item.kind, item.number)
    state = item.state or "unknown"
    merged_on = local_date(item.merged_at, zone)
    if item.merged and merged_on:
        state += f" {merged_on.isoformat()}"
    noun = "Merge request" if item.kind == "mr" else "Issue"
    text = f"{noun} {label}: {item.title} ({state})"
    if item.body:
        text += f". {item.body}"
    kind: SourceKind = "gitlab_mr" if item.kind == "mr" else "gitlab_issue"
    return Finding(text=text, source=Source(kind=kind, label=label, url=item.url), when=merged_on)


def diff_finding(source: Source, diff: GitLabDiff) -> Finding:
    text = f"{source.label} changes {diff.path} ({diff.status})"
    if diff.patch:
        text += f":\n{diff.patch}"
    return Finding(text=text, source=source)


def release_finding(
    repo: str,
    release: GitHubRelease,
    zone: tzinfo = UTC,
    *,
    latest: bool,
    kind: SourceKind = "github_release",
) -> Finding:
    published = local_date(release.published_at, zone)
    text = f"Release {release.tag}"
    if release.name and release.name != release.tag:
        text += f" ({release.name})"
    text += f", published {published.isoformat()}" if published else ", publish date unknown"
    if latest:
        text += ". The latest release"
    if release.body:
        text += f". {release.body}"
    source = Source(kind=kind, label=f"{repo}@{release.tag}", url=release.url)
    return Finding(text=text, source=source, when=published)


def code_finding(snippet: CodeSnippet, kind: SourceKind = "github_code") -> Finding:
    where = f"{snippet.repo}:{snippet.path}" if snippet.repo else snippet.path
    label = f"{where} L{snippet.start_line}-L{snippet.end_line}"
    return Finding(
        text=f"{label}:\n{snippet.code}",
        source=Source(kind=kind, label=label, url=snippet.github_url),
        snippet=snippet,
    )
