"""The read-only tools the agent may call for one team. Each result is a list of findings, every
one with the Source a person can check: a meeting moment, a Jira key or a GitHub item."""

from collections.abc import Callable, Coroutine
from datetime import date
from typing import Any, Literal

import anyio
from pydantic import BaseModel

from brain.config import Settings
from brain.github import GitHubItem, GitHubReader, GitHubRelease
from brain.integrations import ToolRefused
from brain.jira import JiraConfig, JiraIssue, JiraReader, root_cause
from brain.memory import MeetingMemory
from brain.report.decisions import terms
from brain.report.extraction import clock
from brain.store import NotFound, Store
from contracts import CodeSnippet, Meeting, Person, Source, TeamSettings

from .code import code_evidence
from .tools import ToolResult, ToolSpec

TaskStatus = Literal["open", "overdue", "all"]

MEMORY_HITS = 6
LIST_LIMIT = 10
TASK_LIMIT = 15
DEFAULT_TIMEOUT = 10.0


class Finding(BaseModel):
    text: str
    source: Source
    when: date | None = None
    snippet: CodeSnippet | None = None  # code: the lines `text` shows, copied from the file


QUERY = {"query": {"type": "string", "description": "Short search text: the topic's key words."}}

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
            description="Search the team's GitHub repository's issues (kind issue) or pull "
            "requests (kind pr).",
            parameters={
                "type": "object",
                "properties": {**QUERY, "kind": {"type": "string", "enum": ["issue", "pr"]}},
                "required": ["query"],
            },
        ),
        "GitHub",
    ),
    (
        ToolSpec(
            name="github_read",
            description="Read one issue (kind issue) or pull request (kind pr) of the team's "
            "GitHub repository by number.",
            parameters={
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["issue", "pr"]},
                },
                "required": ["number"],
            },
        ),
        "GitHub",
    ),
    (
        ToolSpec(
            name="github_releases",
            description="The team's GitHub repository's latest releases, newest first, with "
            "their publish dates. A pull request merged after the latest release is merged but "
            "not released yet.",
            parameters={"type": "object", "properties": {}},
        ),
        "GitHub",
    ),
    (
        ToolSpec(
            name="github_code",
            description="Search the code of the team's GitHub repository; returns the matching "
            "lines of the top files, numbered, with links.",
            parameters={"type": "object", "properties": QUERY, "required": ["query"]},
        ),
        "GitHub code",
    ),
]

RELEASE_LIMIT = 5

SOURCE_NAMES = {spec.name: name for spec, name in SPECS}


def jira_reader(settings: Settings, team: TeamSettings, target: Any = None) -> JiraReader | str:
    """A reader for the team's Jira project, or why there is none. A team without its own
    project uses the deployment's JIRA_PROJECT_KEY, as the push flow does (one team per
    deployment)."""
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


def github_reader(settings: Settings, team: TeamSettings, target: Any = None) -> GitHubReader | str:
    """A reader for the team's repository at the team's ref (default branch when unset), or why
    there is none. Like Jira's project, a team without its own repository uses the deployment's
    GITHUB_REPO (one team per deployment)."""
    repo = team.github.repo or settings.github_repo
    if not settings.github_mcp_url:
        return "GitHub is not configured: set GITHUB_MCP_URL"
    if not repo:
        return "GitHub is not configured: no repository is set in workspace settings or GITHUB_REPO"
    try:
        return GitHubReader(repo, target or settings.github_mcp_url, ref=team.github.ref)
    except ValueError as e:
        return f"GitHub is not configured: {e}"


class TeamToolbox:
    """Every tool reads only `team_id`'s data. Unconfigured tools stay on the menu, marked, so a
    question that needs them can say they are unavailable."""

    def __init__(
        self,
        team_id: str,
        asker_id: str,
        store: Store,
        *,
        members: list[Person],
        memory: MeetingMemory | None,
        jira: JiraReader | str,
        github: GitHubReader | str,
        timeout: float = DEFAULT_TIMEOUT,
        today: date | None = None,
    ):
        self.team_id = team_id
        self.asker_id = asker_id
        self.store = store
        self.members = {p.id: p for p in members}
        self.memory = memory
        self.jira = jira
        self.github = github
        self.timeout = timeout
        self.today = today or date.today()
        self._meetings: dict[str, Meeting | None] = {}
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
        }

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in SPECS]

    def unavailable(self, name: str) -> str | None:
        """Why the tool cannot run right now, or None when it can."""
        if name == "search_meetings" and self.memory is None:
            return "Meeting memory is unavailable: no database or embedding model is configured"
        if name.startswith("jira_") and isinstance(self.jira, str):
            return self.jira
        if name.startswith("github_") and isinstance(self.github, str):
            return self.github
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
            found.append(self.at(meeting, chunk.t_start, chunk.text))
        return found

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
            when = day(meeting)
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
        self, query: str | None = None, kind: str | None = None, **_: Any
    ) -> list[Finding]:
        assert isinstance(self.github, GitHubReader)
        if not (query or "").strip():
            return []
        items = await self.github.search(query, "pr" if kind == "pr" else "issue")
        return [github_finding(self.github.full_name, item) for item in items]

    async def github_read(
        self, number: int | None = None, kind: str | None = None, **_: Any
    ) -> list[Finding]:
        assert isinstance(self.github, GitHubReader)
        if number is None:
            return []
        item = await self.github.read(number, "pr" if kind == "pr" else "issue")
        return [github_finding(self.github.full_name, item)]

    async def github_releases(self, **_: Any) -> list[Finding]:
        assert isinstance(self.github, GitHubReader)
        releases = await self.github.releases(RELEASE_LIMIT)
        return [
            release_finding(self.github.full_name, r, latest=i == 0) for i, r in enumerate(releases)
        ]

    async def github_code(self, query: str | None = None, **_: Any) -> list[Finding]:
        assert isinstance(self.github, GitHubReader)
        if not (query or "").strip():
            return []
        return [code_finding(s) for s in await code_evidence(self.github, query or "")]

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
        return Finding(text=text, source=meeting_source(meeting, t), when=day(meeting))

    def name(self, person_id: str | None) -> str | None:
        person = self.members.get(person_id or "")
        return person.name if person else None


def meeting_source(meeting: Meeting, t: float | None) -> Source:
    label = meeting.title if t is None else f"{meeting.title} {clock(t)}"
    return Source(kind="meeting", label=label, meeting_id=meeting.id, t=t)


def day(meeting: Meeting) -> date | None:
    when = meeting.started_at or meeting.scheduled_start
    return when.date() if when else None


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


def github_finding(repo: str, item: GitHubItem) -> Finding:
    label = f"{repo}#{item.number}"
    state = item.state or "unknown"
    if item.merged:
        state = "merged"
        if item.merged_at:
            state += f" {item.merged_at.date().isoformat()}"
    noun = "Pull request" if item.kind == "pr" else "Issue"
    text = f"{noun} {label}: {item.title} ({state})"
    if item.body:
        text += f". {item.body}"
    kind = "github_pr" if item.kind == "pr" else "github_issue"
    when = item.merged_at.date() if item.merged_at else None
    return Finding(text=text, source=Source(kind=kind, label=label, url=item.url), when=when)


def release_finding(repo: str, release: GitHubRelease, *, latest: bool) -> Finding:
    published = release.published_at.date() if release.published_at else None
    text = f"Release {release.tag}"
    if release.name and release.name != release.tag:
        text += f" ({release.name})"
    text += f", published {published.isoformat()}" if published else ", publish date unknown"
    if latest:
        text += ". The latest release"
    if release.body:
        text += f". {release.body}"
    source = Source(kind="github_release", label=f"{repo}@{release.tag}", url=release.url)
    return Finding(text=text, source=source, when=published)


def code_finding(snippet: CodeSnippet) -> Finding:
    label = f"{snippet.path} L{snippet.start_line}-L{snippet.end_line}"
    return Finding(
        text=f"{label}:\n{snippet.code}",
        source=Source(kind="github_code", label=label, url=snippet.github_url),
        snippet=snippet,
    )
