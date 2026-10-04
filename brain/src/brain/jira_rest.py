"""Create Jira issues for approved task drafts on a team's own Jira Cloud site, and read its
project's issues, through its REST API, as the account an admin connected in Settings (an
Atlassian email and API token).

Only https://<name>.atlassian.net is ever called, and redirects are never followed, so a typed
site cannot point the brain at another host."""

import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, SecretStr, field_validator

from brain.auth import signing_secret
from brain.config import Settings
from brain.jira import (
    ISSUE_FIELDS,
    JiraConfig,
    JiraError,
    JiraIssue,
    JiraReader,
    OnCreated,
    TaskPusher,
    describe_parts,
    root_cause,
)
from brain.report import ProcessedMeeting
from brain.sealing import Unsealable, unseal
from brain.store import JiraAccount
from contracts import Person, TaskDraft, TaskPushRequest, TaskPushResult

logger = logging.getLogger(__name__)

SITE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\.atlassian\.net")
PROJECT_KEY = re.compile(r"[A-Z][A-Z0-9_]{1,9}")
NOT_A_SITE = "The Jira site must be a Jira Cloud address like your-team.atlassian.net"
TIMEOUT = 15.0
SUMMARY_MAX = 255  # Jira's limit; a summary is also a single line
DEFAULT_ISSUE_TYPE = "Task"
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
    """What a push needs: the site, the account's email and API token, the project, and the
    issue type chosen when the account was connected (None: the type named Task)."""

    site: str
    email: str
    api_token: SecretStr  # never shown when this is printed or dumped
    project_key: str
    issue_type_id: str | None = None

    @field_validator("site")
    @classmethod
    def _cloud_site(cls, site: str) -> str:
        return cloud_site(site)

    @field_validator("project_key")
    @classmethod
    def _key(cls, key: str) -> str:
        if not PROJECT_KEY.fullmatch(key):  # it goes into request paths
            raise ValueError(f"{key!r} is not a Jira project key")
        return key


def account_access(settings: Settings, account: JiraAccount) -> JiraAccess | None:
    """What the team's connected account reaches its site with; None when its token was sealed
    under an AUTH_SECRET this server no longer has."""
    try:
        return JiraAccess(
            site=account.site,
            email=account.email,
            api_token=unseal(account.sealed_token, signing_secret(settings) or "", account.team_id),
            project_key=account.project,
            issue_type_id=account.issue_type_id,
        )
    except (Unsealable, ValueError):
        return None


def reads_with_account(
    settings: Settings, project: str | None, site: str | None, account: JiraAccount | None
) -> bool:
    """Whether the project the agent reads (its key and site) is the one the team's account was
    connected with, and not one of the demo world's (MOCK_JIRA_PROJECTS): it is then read on
    the account's site with it."""
    return (
        account is not None
        and project is not None
        and project.upper() == account.project.upper()
        and (site is None or site.casefold() == account.site.casefold())
        and not settings.mocks_jira_project(project)
    )


class JiraRejected(RuntimeError):
    """Jira answered with an error. `fields` are the issue fields it named, if any."""

    def __init__(self, status: int, message: str, fields: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.fields = fields or {}


class JiraUnreachable(RuntimeError):
    """The site did not answer."""


class JiraTimedOut(JiraUnreachable):
    """The site took the request and never answered: it may have been carried out."""


def issue_type_for_tasks(project: dict[str, Any]) -> str | None:
    """The id of the issue type a project's tasks are created as: its Task, else its first type
    that is not a sub-task. None when the project lists no such type."""
    types = [
        t
        for t in project.get("issueTypes") or []
        if isinstance(t, dict) and isinstance(t.get("id"), str) and not t.get("subtask")
    ]
    named = [t for t in types if str(t.get("name", "")).casefold() == DEFAULT_ISSUE_TYPE.casefold()]
    chosen = (named or types)[:1]
    return chosen[0]["id"] if chosen else None


class JiraCloud:
    """The few REST calls the brain makes to one site, as the connected account."""

    def __init__(self, access: JiraAccess, *, transport: httpx.AsyncBaseTransport | None = None):
        self.access = access
        self.transport = transport

    async def myself(self) -> dict[str, Any]:
        """The connected account; JiraRejected(401) when the email or token is wrong, (404) when
        the address has no Jira site."""
        return await self._call("GET", "/rest/api/3/myself")

    async def project(self, key: str) -> dict[str, Any]:
        """The project with its issue types. JiraRejected(404) when there is no such project or
        the account cannot see it."""
        return await self._call("GET", f"/rest/api/3/project/{key}")

    async def can_create(self, key: str) -> bool:
        """Whether the account may create issues in the project. True when Jira does not say."""
        answer = await self._call(
            "GET",
            "/rest/api/3/mypermissions",
            params={"projectKey": key, "permissions": "CREATE_ISSUES"},
        )
        permissions = answer.get("permissions") if isinstance(answer, dict) else None
        create = permissions.get("CREATE_ISSUES") if isinstance(permissions, dict) else None
        return not isinstance(create, dict) or create.get("havePermission") is not False

    async def assignable(self, query: str) -> list[dict[str, Any]]:
        """The people who can be assigned in the project and whose email or name starts with the
        query (Jira's own matching), never apps or deactivated accounts."""
        users = await self._call(
            "GET",
            "/rest/api/3/user/assignable/search",
            params={"project": self.access.project_key, "query": query},
        )
        return [
            user
            for user in (users if isinstance(users, list) else [])
            if isinstance(user, dict)
            and isinstance(user.get("accountId"), str)
            and user.get("active", True)
            and user.get("accountType", "atlassian") == "atlassian"
        ]

    async def search(self, jql: str, fields: list[str], limit: int) -> list[dict[str, Any]]:
        """The issues the JQL finds, with these fields (Jira's enhanced search)."""
        body = {"jql": jql, "fields": fields, "maxResults": limit}
        found = await self._call("POST", "/rest/api/3/search/jql", json=body)
        issues = found.get("issues") if isinstance(found, dict) else None
        return [issue for issue in issues or [] if isinstance(issue, dict)]

    async def issue(self, key: str, fields: list[str]) -> dict[str, Any]:
        """One issue with these fields. JiraRejected(404) when there is none the account sees."""
        params = {"fields": ",".join(fields)}
        return await self._call("GET", f"/rest/api/3/issue/{key}", params=params)

    async def create_issue(self, fields: dict[str, Any]) -> str:
        """The new issue's key. JiraTimedOut when Jira never answered: it may exist."""
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
                auth=(self.access.email, self.access.api_token.get_secret_value()),
                headers={"Accept": "application/json"},
                timeout=TIMEOUT,
                follow_redirects=False,  # the token goes to this site and nowhere else
                transport=self.transport,
            ) as client:
                response = await client.request(method, path, **request)
        except httpx.TimeoutException as e:
            if isinstance(e, httpx.ConnectTimeout):
                raise JiraUnreachable(f"Could not reach {site}: {root_cause(e)}") from e
            raise JiraTimedOut(f"{site} did not answer in time") from e
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


class JiraRestReader(JiraReader):
    """What JiraReader reads through the MCP server, read from the connected account's project
    on its own site over the REST API: unfinished issues, a text search and one issue."""

    def __init__(self, access: JiraAccess, *, transport: httpx.AsyncBaseTransport | None = None):
        self.cloud = JiraCloud(access, transport=transport)
        # No MCP server: the reads below go to the site itself.
        self.config = JiraConfig(
            mcp_url="",
            cloud_id=access.site,
            project_key=access.project_key,
            base_url=f"https://{access.site}",
        )

    async def _issue(self, key: str) -> object:
        try:
            return readable(await self.cloud.issue(key, ISSUE_FIELDS))
        except JiraRejected as e:
            if e.status == 404:
                raise JiraError(f"{key} was not found in {self.project}") from None
            raise JiraError(f"Jira answered {e.status}: {e}") from None
        except JiraUnreachable as e:
            raise JiraError(str(e)) from None

    async def _search(self, jql: str, limit: int, fields: list[str]) -> list[JiraIssue]:
        try:
            found = await self.cloud.search(jql, fields, limit)
        except JiraRejected as e:
            raise JiraError(f"Jira answered {e.status}: {e}") from None
        except JiraUnreachable as e:
            raise JiraError(str(e)) from None
        issues = (self.issue(readable(raw)) for raw in found)
        return [issue for issue in issues if issue is not None]


# Atlassian Document Format nodes that end a line of their own.
BLOCKS = {"paragraph", "heading", "listItem", "blockquote", "codeBlock", "rule", "tableRow"}


def readable(raw: dict[str, Any]) -> dict[str, Any]:
    """The issue with its rich-text description as plain text, as the MCP server gives it."""
    fields = raw.get("fields")
    if not isinstance(fields, dict) or not isinstance(fields.get("description"), dict):
        return raw
    text = plain_text(fields["description"]).strip()
    return {**raw, "fields": {**fields, "description": text or None}}


def plain_text(node: Any) -> str:
    """The text of an Atlassian Document Format node, a line per block."""
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return str(node.get("text") or "")
    if node.get("type") == "hardBreak":
        return "\n"
    inner = "".join(plain_text(child) for child in node.get("content") or [])
    return inner + "\n" if node.get("type") in BLOCKS else inner


class JiraRestPusher(TaskPusher):
    """Creates the approved drafts as issues in the team's project on its Jira Cloud site."""

    def __init__(self, access: JiraAccess, *, transport: httpx.AsyncBaseTransport | None = None):
        self.access = access
        self.cloud = JiraCloud(access, transport=transport)

    async def create_all(
        self,
        drafts: list[TaskDraft],
        meeting: ProcessedMeeting,
        request: TaskPushRequest,
        on_created: OnCreated,
    ) -> dict[str, TaskPushResult]:
        """Each draft on its own: one failing never loses the keys of those already created."""
        results: dict[str, TaskPushResult] = {}
        for draft in drafts:
            try:
                results[draft.id] = result = await self.create(draft, meeting, request)
                await on_created(result)
            except JiraTimedOut as e:
                project = self.access.project_key
                error = (
                    f"{e}; the issue may have been created. Look for it in {project} before "
                    "pushing this task again"
                )
                results[draft.id] = TaskPushResult(task_id=draft.id, error=error)
            except JiraUnreachable as e:  # the rest would fail the same way
                for left in drafts:
                    results.setdefault(left.id, TaskPushResult(task_id=left.id, error=str(e)))
                break
            except JiraRejected as e:
                results[draft.id] = TaskPushResult(task_id=draft.id, error=self.refusal(e))
            except Exception:
                logger.exception("pushing task draft %s to Jira failed", draft.id)
                error = "The push failed unexpectedly; see the server log"
                results.setdefault(draft.id, TaskPushResult(task_id=draft.id, error=error))
        return results

    async def create(
        self, draft: TaskDraft, meeting: ProcessedMeeting, request: TaskPushRequest
    ) -> TaskPushResult:
        issue_type = self.access.issue_type_id
        fields: dict[str, Any] = {
            "project": {"key": self.access.project_key},
            "issuetype": {"id": issue_type} if issue_type else {"name": DEFAULT_ISSUE_TYPE},
            "summary": " ".join(draft.title.split())[:SUMMARY_MAX],
            "description": document(describe_parts(draft, meeting, request.approved_by)),
        }
        warnings: list[str] = []
        assignee, warning = await self.assignee(draft, meeting)
        if assignee:
            fields["assignee"] = {"id": assignee}
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
            ids = same_person(await self.cloud.assignable(search), owner)
        except (JiraRejected, JiraUnreachable) as e:
            return None, f"created unassigned: looking up {search} failed: {e}"
        if len(ids) == 1:
            return ids[0], None
        if not ids:
            return None, f"created unassigned: no Jira account matches {search}"
        return None, f"created unassigned: {len(ids)} Jira accounts match {search}"

    def url(self, key: str) -> str:
        return f"https://{self.access.site}/browse/{key}"

    def pushed_url(self, draft: TaskDraft) -> str | None:
        """Only the link saved with the draft: a key from before the account was connected
        belongs to wherever it was pushed then, not to this site."""
        return draft.url

    def refusal(self, error: JiraRejected) -> str:
        if error.status in (401, 403):
            return (
                f"Jira did not accept the connected account ({error.status}); "
                "an admin has to connect Jira again in Settings"
            )
        return f"Jira refused it: {error}"


def same_person(users: list[dict[str, Any]], owner: Person) -> list[str]:
    """Account ids of the found users who are the owner. Jira matches the start of a name or
    email, so "Sam" also finds Samantha: only the same email counts, or the same name when the
    owner has no email. One user whose email Jira hides, found by the owner's email, counts too."""

    def same(a: object, b: str) -> bool:
        return isinstance(a, str) and a.strip().casefold() == b.strip().casefold()

    if owner.email:
        exact = [u for u in users if same(u.get("emailAddress"), owner.email)]
        if not exact and len(users) == 1 and not users[0].get("emailAddress"):
            exact = users
    else:
        exact = [u for u in users if same(u.get("displayName"), owner.name)]
    return list(dict.fromkeys(u["accountId"] for u in exact))


def document(parts: list[tuple[str, str]]) -> dict[str, Any]:
    """An Atlassian Document Format description: a paragraph per part, the quote as a quote."""

    def paragraph(text: str) -> dict[str, Any]:
        return {"type": "paragraph", "content": [{"type": "text", "text": text}]}

    content = [
        {"type": "blockquote", "content": [paragraph(text)]} if kind == "quote" else paragraph(text)
        for kind, text in parts
    ]
    return {"type": "doc", "version": 1, "content": content}
