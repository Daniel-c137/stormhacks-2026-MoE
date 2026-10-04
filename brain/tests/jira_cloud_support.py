"""A stand-in for one Jira Cloud site's REST API, served through an httpx transport."""

import base64
import json
import re

import anyio
import httpx

SITE = "acme.atlassian.net"
EMAIL = "admin@acme.example"
TOKEN = "jira-api-token-for-tests"


TASK = {"id": "10001", "name": "Task", "subtask": False}
SUBTASK = {"id": "10002", "name": "Sub-task", "subtask": True}
STORY = {"id": "10003", "name": "Story", "subtask": False}


class FakeJiraCloud:
    """Answers what the brain calls on https://<site>/rest/api/3: myself, a project and its issue
    types, the caller's permission to create issues, assignable users, issue creation, and the
    reads: an enhanced JQL search (the issues in `issues` of the project the JQL names, whose
    summary or description holds every quoted word when it has text ~, unfinished ones when
    it excludes Done) and one issue. Any other host cannot be reached.

    As Jira does: a user search matches the start of an email or of a word of a name; a summary
    over 255 characters or with a line break, or starting with FAIL, is rejected; so is an issue
    type the project lacks. Fields named in `unsettable` are rejected the way Jira rejects a
    field that is not on the project's create screen.

    To stage trouble: `down` (no connection), `missing` (no such site: Atlassian's 404 page),
    `redirect` (every answer is a 302 there), `slow` (seconds each create takes),
    `crash_on_create` and `time_out_on_create` (the nth create raises)."""

    def __init__(self, *, projects=("DS",), users=(), issue_types=(TASK, SUBTASK, STORY)):
        self.projects = set(projects)
        self.users = [
            {"accountId": "acc-admin", "displayName": "Acme Admin", "emailAddress": EMAIL},
            *users,
        ]
        self.issue_types = list(issue_types)
        self.can_create = True
        self.hide_emails = False
        self.unsettable: set[str] = set()
        self.down = False
        self.missing = False
        self.redirect: str | None = None
        self.slow = 0.0
        self.crash_on_create: int | None = None
        self.time_out_on_create: int | None = None
        self.creates = 0
        self.requests: list[httpx.Request] = []
        self.created: list[dict] = []
        self.issues: list[dict] = []  # as Jira's REST API returns them, rich-text descriptions
        self.searches: list[dict] = []
        self.transport = httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down or request.url.host != SITE or request.url.scheme != "https":
            raise httpx.ConnectError("no route to host", request=request)
        if self.redirect:
            return httpx.Response(302, headers={"Location": self.redirect})
        if self.missing:
            return httpx.Response(404, text="<html>Atlassian Cloud: site not found</html>")
        expected = "Basic " + base64.b64encode(f"{EMAIL}:{TOKEN}".encode()).decode()
        if request.headers.get("authorization") != expected:
            return httpx.Response(401, text="Client must be authenticated to access this resource.")
        path = request.url.path
        if path == "/rest/api/3/myself":
            return httpx.Response(200, json=self.users[0])
        if path.startswith("/rest/api/3/project/"):
            key = path.rsplit("/", 1)[1]
            if key in self.projects:
                project = {"key": key, "name": f"Project {key}", "issueTypes": self.issue_types}
                return httpx.Response(200, json=project)
            message = f"No project could be found with key '{key}'."
            return httpx.Response(404, json={"errorMessages": [message], "errors": {}})
        if path == "/rest/api/3/mypermissions":
            assert request.url.params["permissions"] == "CREATE_ISSUES"
            assert request.url.params["projectKey"] in self.projects
            allowed = {"CREATE_ISSUES": {"key": "CREATE_ISSUES", "havePermission": self.can_create}}
            return httpx.Response(200, json={"permissions": allowed})
        if path == "/rest/api/3/user/assignable/search":
            return httpx.Response(200, json=self.search(request.url.params["query"]))
        if path == "/rest/api/3/search/jql" and request.method == "POST":
            body = json.loads(request.content)
            self.searches.append(body)
            return httpx.Response(200, json={"issues": self.found(body), "isLast": True})
        if path.startswith("/rest/api/3/issue/") and request.method == "GET":
            key = path.rsplit("/", 1)[1]
            for issue in self.issues:
                if issue["key"] == key:
                    return httpx.Response(200, json=issue)
            message = "Issue does not exist or you do not have permission to see it."
            return httpx.Response(404, json={"errorMessages": [message], "errors": {}})
        if path == "/rest/api/3/issue" and request.method == "POST":
            self.creates += 1
            if self.slow:
                await anyio.sleep(self.slow)
            if self.creates == self.crash_on_create:
                raise RuntimeError("something unexpected")
            if self.creates == self.time_out_on_create:
                raise httpx.ReadTimeout("timed out", request=request)
            return self.create(json.loads(request.content)["fields"])
        return httpx.Response(404, json={"errorMessages": ["Not found"], "errors": {}})

    def found(self, body: dict) -> list[dict]:
        jql = body["jql"]
        project = re.search(r'project = "([^"]+)"', jql)
        words = re.search(r'text ~ "([^"]*)"', jql)
        found = []
        for issue in self.issues:
            fields = issue["fields"]
            if project and not issue["key"].startswith(project.group(1) + "-"):
                continue
            if (
                "statusCategory != Done" in jql
                and fields["status"]["statusCategory"]["key"] == "done"
            ):
                continue
            text = f"{fields['summary']} {text_of(fields.get('description') or {'type': 'doc'})}"
            if words and not all(w.casefold() in text.casefold() for w in words.group(1).split()):
                continue
            found.append(issue)
        return found[: body.get("maxResults", 50)]

    def search(self, query: str) -> list[dict]:
        query = query.casefold()
        found = []
        for user in self.users:
            words = [user["displayName"], *user["displayName"].split(), user["emailAddress"]]
            if any(word.casefold().startswith(query) for word in words):
                shown = {
                    k: v for k, v in user.items() if k != "emailAddress" or not self.hide_emails
                }
                found.append({**shown, "accountType": "atlassian", "active": True})
        return found

    def create(self, fields: dict) -> httpx.Response:
        errors = {
            name: f"Field '{name}' cannot be set. It is not on the appropriate screen, or unknown."
            for name in fields
            if name in self.unsettable
        }
        summary = fields["summary"]
        if summary.startswith("FAIL"):
            errors["summary"] = "Summary is not valid."
        elif len(summary) > 255:
            errors["summary"] = "Summary must be less than 255 characters."
        elif "\n" in summary:
            errors["summary"] = "The summary is invalid because it contains newline characters."
        if fields["project"]["key"] not in self.projects:
            errors["project"] = "Specify a valid project ID or key"
        wanted = fields["issuetype"]
        creatable = [t for t in self.issue_types if not t["subtask"]]
        if not any(wanted in ({"id": t["id"]}, {"name": t["name"]}) for t in creatable):
            errors["issuetype"] = "Specify a valid issue type"
        if "assignee" in fields and "assignee" not in errors:
            known = {user["accountId"] for user in self.users}
            if fields["assignee"].get("id") not in known:
                errors["assignee"] = "Specify a valid value for assignee"
        if errors:
            return httpx.Response(400, json={"errorMessages": [], "errors": errors})
        self.created.append(fields)
        key = f"{fields['project']['key']}-{len(self.created)}"
        return httpx.Response(201, json={"id": str(10000 + len(self.created)), "key": key})


def text_of(document: dict) -> str:
    """The plain text of an Atlassian Document Format description, one line per block."""

    def walk(node: dict) -> str:
        if node["type"] == "text":
            return node["text"]
        inner = [walk(child) for child in node.get("content", [])]
        return ("\n" if node["type"] in ("doc", "blockquote") else "").join(inner)

    return walk(document)


def rest_issue(
    key: str,
    summary: str,
    *,
    done: bool = False,
    description: str | None = None,
    assignee: str | None = "Sarah Kim",
) -> dict:
    """An issue as Jira Cloud's REST API returns it: the description in Atlassian Document
    Format."""
    status = (
        {"name": "Done", "statusCategory": {"key": "done"}}
        if done
        else {
            "name": "In Progress",
            "statusCategory": {"key": "indeterminate"},
        }
    )
    fields: dict = {
        "summary": summary,
        "status": status,
        "assignee": {"displayName": assignee} if assignee else None,
        "priority": {"name": "High"},
        "description": None,
    }
    if description is not None:
        paragraph = {"type": "paragraph", "content": [{"type": "text", "text": description}]}
        fields["description"] = {"type": "doc", "version": 1, "content": [paragraph]}
    return {"id": key.rsplit("-", 1)[1], "key": key, "fields": fields}
