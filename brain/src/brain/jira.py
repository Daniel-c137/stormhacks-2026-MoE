"""Create Jira issues for task drafts a person approved, and read unfinished work, through the
Jira MCP server.

The same tool names work against the world's mock and the real server; JIRA_MCP_URL decides.
A team whose admin connected its Jira account pushes through the site's REST API instead
(brain.jira_rest); what a push is, and which drafts it creates, is the same for both.
"""

import json
import re
from collections.abc import Awaitable, Callable
from typing import Literal

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from brain.config import Settings
from brain.integrations import McpReader, McpToolError, ToolRefused, search_words
from brain.report import ProcessedMeeting
from brain.report.extraction import clock
from brain.zones import local_date, team_zone
from contracts import TaskDraft, TaskPushRequest, TaskPushResult


class JiraUnavailable(RuntimeError):
    """Jira is not configured."""


class JiraError(RuntimeError):
    """A Jira read failed."""


class ApprovalRequired(PermissionError):
    """A push without a named approver."""


class JiraConfig(BaseModel):
    mcp_url: str
    cloud_id: str
    project_key: str
    base_url: str | None = None
    issue_type: str = "Task"


def jira_config(settings: Settings) -> JiraConfig:
    cloud_id = settings.jira_cloud_id or settings.jira_base_url
    required = {
        "JIRA_MCP_URL": settings.jira_mcp_url,
        "JIRA_PROJECT_KEY": settings.jira_project_key,
        "JIRA_CLOUD_ID (or JIRA_BASE_URL)": cloud_id,
    }
    if missing := [name for name, value in required.items() if not value]:
        raise JiraUnavailable(f"Jira is not configured: set {', '.join(missing)}")
    return JiraConfig(
        mcp_url=settings.jira_mcp_url,
        cloud_id=cloud_id,
        project_key=settings.jira_project_key,
        base_url=settings.jira_base_url or None,
    )


OnCreated = Callable[[TaskPushResult], Awaitable[None]]
"""Called with each new issue's result as soon as it exists, so its key can be saved at once."""


class TaskPusher:
    """An approved push, wherever the issues are created: which drafts go, and one result each."""

    async def push(
        self,
        meeting: ProcessedMeeting,
        request: TaskPushRequest,
        *,
        on_created: OnCreated | None = None,
    ) -> list[TaskPushResult]:
        """One result per requested draft, in request order. Never raises for one bad draft."""
        if not request.approved_by.strip():
            raise ApprovalRequired("Pushing to Jira needs the name of the person who approved it")
        task_ids = list(dict.fromkeys(request.task_ids))
        if request.destination != "jira":
            error = f"Pushing to {request.destination} is not supported yet"
            return [TaskPushResult(task_id=task_id, error=error) for task_id in task_ids]

        drafts = {task.id: task for task in meeting.report.tasks}
        results: dict[str, TaskPushResult] = {}
        to_create: list[TaskDraft] = []
        for task_id in task_ids:
            draft = drafts.get(task_id)
            if draft is None:
                results[task_id] = TaskPushResult(task_id=task_id, error="No such task draft")
            elif not draft.include:
                results[task_id] = TaskPushResult(task_id=task_id, error="Draft is excluded")
            elif draft.key:  # pushed before, never again
                results[task_id] = TaskPushResult(
                    task_id=task_id, key=draft.key, url=self.pushed_url(draft)
                )
            else:
                to_create.append(draft)

        if to_create:
            results |= await self.create_all(to_create, meeting, request, on_created or _nothing)
        return [results[task_id] for task_id in task_ids]

    async def create_all(
        self,
        drafts: list[TaskDraft],
        meeting: ProcessedMeeting,
        request: TaskPushRequest,
        on_created: OnCreated,
    ) -> dict[str, TaskPushResult]:
        """A result for every draft: its new key, or why it was not created. `on_created` is
        awaited for each issue as it is created."""
        raise NotImplementedError

    def url(self, key: str) -> str | None:
        """Where the issue can be opened, when the site's address is known."""
        raise NotImplementedError

    def pushed_url(self, draft: TaskDraft) -> str | None:
        """The link of a draft pushed earlier: the one saved with it, else this site's."""
        return draft.url or self.url(draft.key or "")


async def _nothing(result: TaskPushResult) -> None:
    return None


class JiraPusher(TaskPusher):
    def __init__(self, config: JiraConfig, *, target: str | MCPServer | None = None):
        self.config = config
        self.target = target or config.mcp_url

    async def create_all(
        self,
        drafts: list[TaskDraft],
        meeting: ProcessedMeeting,
        request: TaskPushRequest,
        on_created: OnCreated,
    ) -> dict[str, TaskPushResult]:
        results: dict[str, TaskPushResult] = {}
        try:
            async with Client(self.target) as client:
                for draft in drafts:
                    results[draft.id] = result = await self.create(client, draft, meeting, request)
                    if result.key:
                        await on_created(result)
        except Exception as e:  # transport failure: report it per draft, keep what was created
            error = f"Jira MCP call failed: {root_cause(e)}"
            for draft in drafts:
                results.setdefault(draft.id, TaskPushResult(task_id=draft.id, error=error))
        return results

    async def create(
        self,
        client: Client,
        draft: TaskDraft,
        meeting: ProcessedMeeting,
        request: TaskPushRequest,
    ) -> TaskPushResult:
        assignee, warning = await self.assignee(client, draft, meeting)
        result = await client.call_tool(
            "createJiraIssue",
            {
                "cloudId": self.config.cloud_id,
                "projectKey": self.config.project_key,
                "issueTypeName": self.config.issue_type,
                "summary": draft.title,
                "description": describe(draft, meeting, request.approved_by),
                "assignee_account_id": assignee,
                "additional_fields": {"duedate": draft.due.isoformat()} if draft.due else None,
            },
        )
        if result.is_error:
            return TaskPushResult(task_id=draft.id, error=text_of(result) or "Jira refused it")
        if not (key := issue_key(result)):
            return TaskPushResult(task_id=draft.id, error="Jira returned no issue key")
        return self.pushed(draft.id, key, warning)

    async def assignee(
        self, client: Client, draft: TaskDraft, meeting: ProcessedMeeting
    ) -> tuple[str | None, str | None]:
        """The owner's Jira account id, or None and a warning saying why the issue is unassigned.
        Only an owner who is a meeting member is looked up, by email, else by name."""
        owner = next((p for p in meeting.members if p.id == draft.owner_id), None)
        if owner is None:
            return None, None
        search = owner.email or owner.name
        result = await client.call_tool(
            "lookupJiraAccountId", {"cloudId": self.config.cloud_id, "searchString": search}
        )
        if result.is_error:
            reason = text_of(result) or "Jira refused it"
            return None, f"created unassigned: looking up {search} failed: {reason}"
        ids = account_ids(result)
        if len(ids) == 1:
            return ids[0], None
        if not ids:
            return None, f"created unassigned: no Jira account matches {search}"
        return None, f"created unassigned: {len(ids)} Jira accounts match {search}"

    def pushed(self, task_id: str, key: str, warning: str | None = None) -> TaskPushResult:
        return TaskPushResult(task_id=task_id, key=key, url=self.url(key), warning=warning)

    def url(self, key: str) -> str | None:
        base = self.config.base_url
        return f"{base.rstrip('/')}/browse/{key}" if base else None


class JiraIssue(BaseModel):
    key: str
    summary: str
    status: str | None = None
    assignee: str | None = None
    priority: str | None = None
    description: str | None = None
    done: bool = False
    url: str | None = None


ISSUE_FIELDS = ["summary", "status", "assignee", "priority", "description"]
MAX_DESCRIPTION = 400


class JiraReader:
    """Read-only searches and reads of the configured project through the Jira MCP server. Every
    call goes through the read allowlist, so a write tool is refused before Jira is reached.
    brain.jira_rest.JiraRestReader reads a connected account's project the same way over the
    site's REST API."""

    def __init__(self, config: JiraConfig, *, target: str | MCPServer | None = None):
        self.config = config
        self.target = target or config.mcp_url
        self.reader = McpReader("jira", self.target)

    @property
    def project(self) -> str:
        return re.sub(r'["\\]', "", self.config.project_key)

    async def unfinished(self, limit: int = 10) -> list[JiraIssue]:
        """The project's issues not done yet, most recently updated first."""
        jql = f'project = "{self.project}" AND statusCategory != Done ORDER BY updated DESC'
        issues = await self._search(jql, limit, ["summary", "status", "assignee"])
        return [issue for issue in issues if not issue.done][:limit]

    async def search(self, text: str, limit: int = 8) -> list[JiraIssue]:
        """The project's issues whose text matches, done or not, most recently updated first.
        Only words reach the JQL, so the text cannot widen the search beyond the project."""
        words = search_words(text)
        if not words:
            return []
        jql = f'project = "{self.project}" AND text ~ "{words}" ORDER BY updated DESC'
        return (await self._search(jql, limit, ISSUE_FIELDS))[:limit]

    async def get(self, key: str) -> JiraIssue:
        """One issue of the project. A key from another project is refused."""
        key = key.strip().upper()
        if not re.fullmatch(rf"{re.escape(self.project.upper())}-\d+", key):
            raise JiraError(f"{key} is not an issue of the {self.project} project")
        issue = self.issue(await self._issue(key))
        if issue is None:
            raise JiraError(f"Jira returned no issue for {key}")
        return issue

    async def _issue(self, key: str) -> object:
        data = await self._call(
            "getJiraIssue",
            {"cloudId": self.config.cloud_id, "issueIdOrKey": key, "fields": ISSUE_FIELDS},
        )
        if isinstance(data, dict) and "key" not in data and isinstance(data.get("result"), dict):
            data = data["result"]
        return data

    async def _search(self, jql: str, limit: int, fields: list[str]) -> list[JiraIssue]:
        data = await self._call(
            "searchJiraIssuesUsingJql",
            {"cloudId": self.config.cloud_id, "jql": jql, "fields": fields, "maxResults": limit},
        )
        issues = (self.issue(raw) for raw in raw_issues(data))
        return [issue for issue in issues if issue is not None]

    async def _call(self, tool: str, arguments: dict) -> object:
        try:
            return await self.reader.call(tool, arguments)
        except ToolRefused:
            raise
        except McpToolError as e:
            raise JiraError(str(e)) from e
        except Exception as e:
            raise JiraError(f"Jira MCP call failed: {root_cause(e)}") from e

    def issue(self, raw: object) -> JiraIssue | None:
        if not isinstance(raw, dict) or not isinstance(key := raw.get("key"), str):
            return None
        fields = raw.get("fields") if isinstance(raw.get("fields"), dict) else {}
        status = fields.get("status") if isinstance(fields.get("status"), dict) else {}
        category = status.get("statusCategory")
        assignee = fields.get("assignee") if isinstance(fields.get("assignee"), dict) else {}
        priority = fields.get("priority") if isinstance(fields.get("priority"), dict) else {}
        description = fields.get("description")
        if isinstance(description, str) and description.strip():
            description = description.strip()[:MAX_DESCRIPTION]
        else:
            description = None  # Atlassian also sends rich-text documents; only text is kept
        url = f"{self.config.base_url.rstrip('/')}/browse/{key}" if self.config.base_url else None
        return JiraIssue(
            key=key,
            summary=str(fields.get("summary") or "").strip(),
            status=status.get("name"),
            assignee=assignee.get("displayName"),
            priority=priority.get("name"),
            description=description,
            done=isinstance(category, dict) and category.get("key") == "done",
            url=url,
        )


def raw_issues(data: object) -> list:
    if not isinstance(data, dict):
        return []
    for candidate in (data, data.get("result")):
        if isinstance(candidate, dict) and isinstance(candidate.get("issues"), list):
            return candidate["issues"]
    return []


def apply_results(tasks: list[TaskDraft], results: list[TaskPushResult]) -> list[TaskDraft]:
    """Record new keys, and where their issues open, on the drafts that were just created."""
    created = {r.task_id: r for r in results if r.key}
    return [
        task.model_copy(
            update={"key": created[task.id].key, "url": created[task.id].url, "jira_status": "todo"}
        )
        if task.id in created and not task.key
        else task
        for task in tasks
    ]


def describe_parts(
    draft: TaskDraft, meeting: ProcessedMeeting, approved_by: str
) -> list[tuple[Literal["text", "quote"], str]]:
    """An issue's description, a paragraph at a time: what the draft says, the meeting and
    moment it came from, the words quoted, the owner named, and who approved it."""
    day = local_date(meeting.started_at, team_zone(meeting.timezone))
    on = f" ({day.isoformat()})" if day else ""
    at = f" at {clock(draft.t)}" if draft.t is not None else ""
    names = {person.id: person.name for person in meeting.members}
    parts: list[tuple[Literal["text", "quote"], str]] = []
    if draft.description:
        parts.append(("text", draft.description))
    source = f'From the meeting "{meeting.title}"{on}{at}' + (":" if draft.quote else ".")
    parts.append(("text", source))
    if draft.quote:
        parts.append(("quote", draft.quote))
    if draft.owner_id in names:
        parts.append(("text", f"Owner named in the meeting: {names[draft.owner_id]}"))
    parts.append(("text", f"Approved for Jira by {approved_by.strip()}."))
    return parts


def describe(draft: TaskDraft, meeting: ProcessedMeeting, approved_by: str) -> str:
    parts = describe_parts(draft, meeting, approved_by)
    return "\n\n".join(f"> {text}" if kind == "quote" else text for kind, text in parts)


def issue_key(result: CallToolResult) -> str | None:
    data = result.structured_content
    if not isinstance(data, dict):
        try:
            data = json.loads(text_of(result))
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    for candidate in (data, data.get("result"), data.get("issue")):
        if isinstance(candidate, dict) and isinstance(candidate.get("key"), str):
            return candidate["key"]
    return None


def account_ids(result: CallToolResult) -> list[str]:
    """Distinct account ids from a lookupJiraAccountId result.

    Assumed shape, unverified against Atlassian's server: a list of users, or an object with
    that list under "users" or "result" (possibly nested once), each user carrying "accountId"
    (or "account_id"). Anything else counts as no match."""
    data = result.structured_content
    if data is None:
        try:
            data = json.loads(text_of(result))
        except ValueError:
            return []
    for _ in range(3):
        if not isinstance(data, dict):
            break
        data = data.get("users", data.get("result"))
    if not isinstance(data, list):
        return []
    ids = (
        user.get("accountId") or user.get("account_id") for user in data if isinstance(user, dict)
    )
    return list(dict.fromkeys(i for i in ids if isinstance(i, str) and i))


def text_of(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


def site_host(site: str | None) -> str | None:
    """A Jira site as the board links to it, https://<site>/browse/KEY: the host and any path,
    without a scheme or trailing slash ("https://acme.atlassian.net/" is acme.atlassian.net).
    Blank is None."""
    if site is None:
        return None
    clean = re.sub(r"^[a-z][a-z0-9+.-]*://", "", site.strip(), flags=re.IGNORECASE).rstrip("/")
    return clean or None


def root_cause(error: BaseException) -> str:
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return str(error) or type(error).__name__
