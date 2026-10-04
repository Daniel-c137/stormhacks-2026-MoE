"""Create Jira issues for approved task drafts on a team's own Jira Cloud site, through its REST
API, as the account an admin connected in Settings (an Atlassian email and API token).

Only https://<name>.atlassian.net is ever called, so a typed site cannot point the brain at
another host."""

import re
from typing import Any

import httpx
from pydantic import BaseModel, field_validator

from brain.jira import TaskPusher, describe_parts, root_cause
from brain.report import ProcessedMeeting
from contracts import TaskDraft, TaskPushRequest, TaskPushResult

SITE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\.atlassian\.net")
NOT_A_SITE = "The Jira site must be a Jira Cloud address like your-team.atlassian.net"
TIMEOUT = 15.0
# Fields a project may not have on its create screen; the issue is created without them.
OPTIONAL_FIELDS = {"duedate": "a due date", "assignee": "an assignee"}


def cloud_site(site: str) -> str:
    """The site's host, e.g. acme.atlassian.net, from that or a pasted link to it. ValueError
    for anything that is not a Jira Cloud site."""
    host = re.sub(r"^https?://", "", site.strip().lower()).split("/", 1)[0]
    if not SITE.fullmatch(host):
        raise ValueError(NOT_A_SITE)
    return host


class JiraAccess(BaseModel):
    """What a push needs: the site, the account's email and API token, and the project."""

    site: str
    email: str
    api_token: str
    project_key: str
    issue_type: str = "Task"

    @field_validator("site")
    @classmethod
    def _cloud_site(cls, site: str) -> str:
        return cloud_site(site)


class JiraRejected(RuntimeError):
    """Jira answered with an error. `fields` are the issue fields it named, if any."""

    def __init__(self, status: int, message: str, fields: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.fields = fields or {}


class JiraUnreachable(RuntimeError):
    """The site did not answer."""


class JiraCloud:
    """The few REST calls the brain makes to one site, as the connected account."""

    def __init__(self, access: JiraAccess, *, transport: httpx.AsyncBaseTransport | None = None):
        self.access = access
        self.transport = transport

    async def myself(self) -> dict[str, Any]:
        """The connected account; JiraRejected(401) when the email or token is wrong."""
        return await self._call("GET", "/rest/api/3/myself")

    async def project(self, key: str) -> dict[str, Any]:
        """JiraRejected(404) when there is no such project or the account cannot see it."""
        return await self._call("GET", f"/rest/api/3/project/{key}")

    async def assignable(self, query: str) -> list[str]:
        """Account ids of the people who can be assigned in the project and match the email or
        name, never apps or deactivated accounts."""
        users = await self._call(
            "GET",
            "/rest/api/3/user/assignable/search",
            params={"project": self.access.project_key, "query": query},
        )
        ids = [
            user.get("accountId")
            for user in (users if isinstance(users, list) else [])
            if isinstance(user, dict)
            and user.get("active", True)
            and user.get("accountType", "atlassian") == "atlassian"
        ]
        return list(dict.fromkeys(i for i in ids if isinstance(i, str) and i))

    async def create_issue(self, fields: dict[str, Any]) -> str:
        """The new issue's key."""
        created = await self._call("POST", "/rest/api/3/issue", json={"fields": fields})
        key = created.get("key") if isinstance(created, dict) else None
        if not isinstance(key, str) or not key:
            raise JiraRejected(502, "Jira returned no issue key")
        return key

    async def _call(self, method: str, path: str, **request: Any) -> Any:
        site = self.access.site
        try:
            async with httpx.AsyncClient(
                base_url=f"https://{site}",
                auth=(self.access.email, self.access.api_token),
                headers={"Accept": "application/json"},
                timeout=TIMEOUT,
                transport=self.transport,
            ) as client:
                response = await client.request(method, path, **request)
        except httpx.HTTPError as e:
            raise JiraUnreachable(f"Could not reach {site}: {root_cause(e)}") from e
        if response.is_success:
            try:
                return response.json()
            except ValueError:
                raise JiraRejected(502, f"{site} did not answer as Jira does") from None
        raise rejected(response)


def rejected(response: httpx.Response) -> JiraRejected:
    """Jira's own reason when it gives one. The request, and so the token, is never quoted."""
    try:
        body = response.json()
    except ValueError:
        body = None
    messages: list[str] = []
    fields: dict[str, str] = {}
    if isinstance(body, dict):
        messages = [m for m in body.get("errorMessages") or [] if isinstance(m, str)]
        errors = body.get("errors")
        if isinstance(errors, dict):
            fields = {str(k): str(v) for k, v in errors.items()}
    reason = " ".join([*messages, *fields.values()]) or f"HTTP {response.status_code}"
    return JiraRejected(response.status_code, reason[:300], fields)


class JiraRestPusher(TaskPusher):
    """Creates the approved drafts as issues in the team's project on its Jira Cloud site."""

    def __init__(self, access: JiraAccess, *, transport: httpx.AsyncBaseTransport | None = None):
        self.access = access
        self.cloud = JiraCloud(access, transport=transport)

    async def create_all(
        self, drafts: list[TaskDraft], meeting: ProcessedMeeting, request: TaskPushRequest
    ) -> dict[str, TaskPushResult]:
        results: dict[str, TaskPushResult] = {}
        for draft in drafts:
            try:
                results[draft.id] = await self.create(draft, meeting, request)
            except JiraUnreachable as e:  # the rest would fail the same way
                for left in drafts:
                    results.setdefault(left.id, TaskPushResult(task_id=left.id, error=str(e)))
                break
            except JiraRejected as e:
                results[draft.id] = TaskPushResult(task_id=draft.id, error=self.refusal(e))
        return results

    async def create(
        self, draft: TaskDraft, meeting: ProcessedMeeting, request: TaskPushRequest
    ) -> TaskPushResult:
        fields: dict[str, Any] = {
            "project": {"key": self.access.project_key},
            "issuetype": {"name": self.access.issue_type},
            "summary": draft.title,
            "description": document(describe_parts(draft, meeting, request.approved_by)),
        }
        warnings: list[str] = []
        assignee, warning = await self.assignee(draft, meeting)
        if assignee:
            fields["assignee"] = {"accountId": assignee}
        if warning:
            warnings.append(warning)
        if draft.due:
            fields["duedate"] = draft.due.isoformat()
        try:
            key = await self.cloud.create_issue(fields)
        except JiraRejected as e:
            dropped = [name for name in e.fields if name in fields and name in OPTIONAL_FIELDS]
            if e.status != 400 or not dropped or len(dropped) != len(e.fields):
                raise
            for name in dropped:
                del fields[name]
                warnings.append(
                    f"created without {OPTIONAL_FIELDS[name]}: this Jira project does not take one"
                )
            key = await self.cloud.create_issue(fields)
        return TaskPushResult(
            task_id=draft.id,
            key=key,
            url=self.url(key),
            warning="; ".join(warnings) or None,
        )

    async def assignee(
        self, draft: TaskDraft, meeting: ProcessedMeeting
    ) -> tuple[str | None, str | None]:
        """The owner's Jira account id, or None and a warning saying why the issue is unassigned.
        Only an owner who is a meeting member is looked up, by email, else by name."""
        owner = next((p for p in meeting.members if p.id == draft.owner_id), None)
        if owner is None:
            return None, None
        search = owner.email or owner.name
        try:
            ids = await self.cloud.assignable(search)
        except JiraRejected as e:
            return None, f"created unassigned: looking up {search} failed: {e}"
        if len(ids) == 1:
            return ids[0], None
        if not ids:
            return None, f"created unassigned: no Jira account matches {search}"
        return None, f"created unassigned: {len(ids)} Jira accounts match {search}"

    def url(self, key: str) -> str:
        return f"https://{self.access.site}/browse/{key}"

    def refusal(self, error: JiraRejected) -> str:
        if error.status in (401, 403):
            return (
                f"Jira did not accept the connected account ({error.status}); "
                "an admin has to connect Jira again in Settings"
            )
        return f"Jira refused it: {error}"


def document(parts: list[tuple[str, str]]) -> dict[str, Any]:
    """An Atlassian Document Format description: a paragraph per part, the quote as a quote."""

    def paragraph(text: str) -> dict[str, Any]:
        return {"type": "paragraph", "content": [{"type": "text", "text": text}]}

    content = [
        {"type": "blockquote", "content": [paragraph(text)]} if kind == "quote" else paragraph(text)
        for kind, text in parts
    ]
    return {"type": "doc", "version": 1, "content": content}
