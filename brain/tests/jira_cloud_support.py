"""A stand-in for one Jira Cloud site's REST API, served through an httpx transport."""

import base64
import json

import httpx

SITE = "acme.atlassian.net"
EMAIL = "admin@acme.example"
TOKEN = "jira-api-token-for-tests"


class FakeJiraCloud:
    """Answers what the brain calls on https://<site>/rest/api/3: myself, a project, assignable
    users and issue creation. Any other host cannot be reached. A summary starting with FAIL is
    rejected the way Jira rejects an invalid field; fields named in `unsettable` are rejected
    the way Jira rejects a field that is not on the project's create screen."""

    def __init__(self, *, projects=("DS",), users=()):
        self.projects = set(projects)
        self.users = [
            {"accountId": "acc-admin", "displayName": "Acme Admin", "emailAddress": EMAIL},
            *users,
        ]
        self.unsettable: set[str] = set()
        self.down = False
        self.requests: list[httpx.Request] = []
        self.created: list[dict] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down or request.url.host != SITE or request.url.scheme != "https":
            raise httpx.ConnectError("no route to host", request=request)
        expected = "Basic " + base64.b64encode(f"{EMAIL}:{TOKEN}".encode()).decode()
        if request.headers.get("authorization") != expected:
            return httpx.Response(401, text="Client must be authenticated to access this resource.")
        path = request.url.path
        if path == "/rest/api/3/myself":
            return httpx.Response(200, json=self.users[0])
        if path.startswith("/rest/api/3/project/"):
            key = path.rsplit("/", 1)[1]
            if key in self.projects:
                return httpx.Response(200, json={"key": key, "name": f"Project {key}"})
            message = f"No project could be found with key '{key}'."
            return httpx.Response(404, json={"errorMessages": [message], "errors": {}})
        if path == "/rest/api/3/user/assignable/search":
            query = request.url.params["query"].casefold()
            found = [
                {**user, "accountType": "atlassian", "active": True}
                for user in self.users
                if query in user["emailAddress"].casefold()
                or query in user["displayName"].casefold()
            ]
            return httpx.Response(200, json=found)
        if path == "/rest/api/3/issue" and request.method == "POST":
            return self.create(json.loads(request.content)["fields"])
        return httpx.Response(404, json={"errorMessages": ["Not found"], "errors": {}})

    def create(self, fields: dict) -> httpx.Response:
        errors = {
            name: f"Field '{name}' cannot be set. It is not on the appropriate screen, or unknown."
            for name in fields
            if name in self.unsettable
        }
        if fields["summary"].startswith("FAIL"):
            errors["summary"] = "Summary is not valid."
        if fields["project"]["key"] not in self.projects:
            errors["project"] = "Specify a valid project ID or key"
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
