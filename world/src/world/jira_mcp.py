"""Mock Jira MCP server over mock-data/jira/issues.json, with writes journaled to the overlay.

Tool names follow Atlassian's MCP server; results take Atlassian's REST shapes as far as the brain
reads them (see brain/src/brain/jira.py). Names, parameters and result shapes are not yet
verified against Atlassian's server. The mock serves one site, so any cloudId is accepted.
Done means merged to main; production deploys only from release tags.
"""

import re
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .config import Settings
from .overlay import JournalEntry, JournalOverlay, WriteOp
from .snapshot import mock_records

server = MCPServer("jira")
READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)

Issue = dict[str, Any]
MAX_RESULTS = 100
DEFAULT_RESULTS = 50
STATUS_CATEGORIES = {"new": "To Do", "indeterminate": "In Progress", "done": "Done"}
_write_lock = threading.Lock()  # a write reads the state, checks it and journals as one step


# The world: the mock data with the journal replayed over it


@dataclass
class World:
    issues: list[Issue]
    site: str  # e.g. https://dropsubs.atlassian.net
    projects: list[dict[str, str]]  # {"key", "name"}
    statuses: list[dict[str, Any]]  # the workflow: every status the data uses, by id
    accounts: list[dict[str, Any]]  # every assignee and reporter, by name
    issue_types: list[str]
    priorities: list[str]

    def find(self, id_or_key: str) -> Issue:
        wanted = id_or_key.strip().upper()
        for issue in self.issues:
            if wanted in (issue["key"].upper(), issue["id"]):
                return issue
        raise ToolError(
            f"Issue {id_or_key} does not exist or you do not have permission to see it."
        )

    def account(self, account_id: str) -> dict[str, Any]:
        for account in self.accounts:
            if account["accountId"] == account_id:
                return dict(account)
        raise ToolError(f"Specify a valid value for assignee: no account {account_id!r}")


def load_world() -> World:
    """The mock data with every journaled Jira write applied, in order."""
    issues = mock_records("jira", "issues")
    site = issues[0]["url"].split("/browse/")[0]
    projects = unique(i["fields"]["project"] for i in issues)
    statuses = sorted(unique(i["fields"]["status"] for i in issues), key=lambda s: int(s["id"]))
    people = (i["fields"][role] for i in issues for role in ("assignee", "reporter"))
    accounts = sorted(unique(people), key=lambda a: a["displayName"])
    world = World(
        issues=issues,
        site=site,
        projects=[{"key": p["key"], "name": p["name"]} for p in projects],
        statuses=statuses,
        accounts=accounts,
        issue_types=[t["name"] for t in unique(i["fields"]["issuetype"] for i in issues)],
        priorities=[p["name"] for p in unique(i["fields"]["priority"] for i in issues)],
    )
    for entry in JournalOverlay.configured().journal():
        if entry.server == "jira":
            apply(world, entry)
    return world


def unique(items) -> list[dict[str, Any]]:
    """The distinct objects, skipping nulls, in first-seen order."""
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        if item:
            seen.setdefault(repr(sorted(item.items())), item)
    return list(seen.values())


def write(tool: str, op: WriteOp, target: str, arguments: dict[str, Any]) -> Issue:
    """Apply a write to the current world and journal it; an invalid write raises before it is
    journaled. `target` is the issue's key or id, or the project's key for a create. Returns
    the issue written."""
    with _write_lock:
        world = load_world()
        if op == "create_issue":
            numbers = [
                int(i["key"].rsplit("-", 1)[1])
                for i in world.issues
                if i["fields"]["project"]["key"] == target
            ]
            target = f"{target}-{max(numbers, default=0) + 1}"
        else:
            target = world.find(target)["key"]
        entry = JournalEntry.now("jira", tool, op, target, arguments)
        issue = apply(world, entry)
        JournalOverlay.configured().record(entry)
        return issue


def apply(world: World, entry: JournalEntry) -> Issue:
    args = entry.arguments
    if entry.op == "create_issue":
        return create(world, entry.target, entry.at, args)
    issue = world.find(entry.target)
    if entry.op == "edit_issue":
        set_fields(world, issue, args["fields"])
    elif entry.op == "transition_issue":
        transition(world, issue, args["transition"], entry.at)
    elif entry.op == "add_comment":
        comment(world, issue, args["commentBody"], entry.at)
    issue["fields"]["updated"] = stamp(entry.at)
    return issue


def create(world: World, key: str, at: datetime, args: dict[str, Any]) -> Issue:
    project = next((p for p in world.projects if p["key"] == args["projectKey"]), None)
    if project is None:
        raise ToolError(f"Specify a valid project ID or key: no project {args['projectKey']!r}")
    issue_type = args["issueTypeName"]
    if issue_type not in world.issue_types:
        raise ToolError(
            f"Specify a valid issue type: {issue_type!r} is not one of "
            f"{', '.join(world.issue_types)}"
        )
    number = int(key.rsplit("-", 1)[1])
    initial = next((s for s in world.statuses if s["name"] == "To Do"), world.statuses[0])
    issue: Issue = {
        "id": str(10000 + number),
        "key": key,
        "self": f"{world.site}/rest/api/3/issue/{key}",
        "url": f"{world.site}/browse/{key}",
        "fields": {
            "summary": "",
            "description": None,
            "issuetype": {"name": issue_type},
            "project": dict(project),
            "status": dict(initial),
            "priority": {"name": "Medium"},
            "assignee": None,
            "reporter": None,  # the mock has no signed-in user
            "labels": [],
            "created": stamp(at),
            "updated": stamp(at),
            "duedate": None,
            "sprint": None,
            "parent": None,
            "comment": {"comments": [], "total": 0},
        },
        "changelog": {"histories": []},
    }
    fields: dict[str, Any] = {"summary": args["summary"], "description": args.get("description")}
    if args.get("assignee_account_id"):
        fields["assignee"] = {"accountId": args["assignee_account_id"]}
    set_fields(world, issue, {**(args.get("additional_fields") or {}), **fields})
    world.issues.append(issue)
    return issue


def set_fields(world: World, issue: Issue, fields: dict[str, Any]) -> None:
    """Validate every field before changing any, as Jira refuses the whole edit."""
    if not fields:
        raise ToolError("Specify at least one field to set")
    changes: dict[str, Any] = {}
    for name, value in fields.items():
        if name == "summary":
            if not isinstance(value, str) or not value.strip():
                raise ToolError("You must specify a summary of the issue.")
            changes[name] = value.strip()
        elif name == "description":
            if value is not None and not isinstance(value, str):
                raise ToolError("Specify the description as plain text")
            changes[name] = value
        elif name == "assignee":
            account_id = value.get("accountId") if isinstance(value, dict) else None
            changes[name] = world.account(account_id) if account_id else None
        elif name == "priority":
            priority = value.get("name") if isinstance(value, dict) else value
            if priority not in world.priorities:
                raise ToolError(f"Specify a valid priority: one of {', '.join(world.priorities)}")
            changes[name] = {"name": priority}
        elif name == "labels":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ToolError("Specify labels as a list of strings")
            changes[name] = value
        elif name == "duedate":
            try:
                changes[name] = date.fromisoformat(value).isoformat() if value else None
            except (TypeError, ValueError):
                raise ToolError(f"Specify the due date as YYYY-MM-DD, not {value!r}") from None
        elif name == "parent":
            parent = value.get("key") if isinstance(value, dict) else None
            changes[name] = {"key": world.find(parent)["key"]} if parent else None
        elif name == "status":
            raise ToolError("Field 'status' cannot be set: use transitionJiraIssue")
        else:
            raise ToolError(
                f"Field '{name}' cannot be set. It is not on the appropriate screen, or unknown."
            )
    issue["fields"].update(changes)


def transitions(world: World, issue: Issue) -> list[dict[str, Any]]:
    """The simplified workflow: from any status to any other. A transition's id is its target
    status's id."""
    current = issue["fields"]["status"]["id"]
    return [
        {"id": s["id"], "name": s["name"], "to": dict(s), "isAvailable": True}
        for s in world.statuses
        if s["id"] != current
    ]


def transition(world: World, issue: Issue, chosen: dict[str, str], at: datetime) -> None:
    wanted = str(chosen.get("id", ""))
    move = next((t for t in transitions(world, issue) if t["id"] == wanted), None)
    if move is None:
        raise ToolError(f"Transition id {wanted!r} is not valid for {issue['key']}")
    before = issue["fields"]["status"]["name"]
    issue["fields"]["status"] = dict(move["to"])
    histories = [h for i in world.issues for h in i.get("changelog", {}).get("histories", [])]
    next_id = max((int(h["id"]) for h in histories), default=9999) + 1
    issue.setdefault("changelog", {"histories": []})["histories"].append(
        {
            "id": str(next_id),
            "created": stamp(at),
            "items": [{"field": "status", "fromString": before, "toString": move["name"]}],
        }
    )


def comment(world: World, issue: Issue, body: str, at: datetime) -> None:
    if not body.strip():
        raise ToolError("Comment body can not be empty!")
    total = sum(i["fields"]["comment"]["total"] for i in world.issues)
    thread = issue["fields"]["comment"]
    thread["comments"].append(
        {"id": str(10001 + total), "body": body, "created": stamp(at), "updated": stamp(at)}
    )
    thread["total"] = len(thread["comments"])


def stamp(at: datetime) -> str:
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


def shaped(issue: Issue, fields: list[str] | None) -> Issue:
    """The issue with only the fields asked for; all of them when none or *all are asked for."""
    if not fields or {"*all", "*navigable"} & set(fields):
        return issue
    return {**issue, "fields": {k: v for k, v in issue["fields"].items() if k in fields}}


# JQL


class JqlError(ToolError):
    pass


SUPPORTED = (
    "This server supports clauses joined by AND on project, key, status and statusCategory "
    '(= and !=), assignee (=, !=, IS EMPTY, IS NOT EMPTY) and text ~ "words", then '
    "ORDER BY created, updated, key or duedate (ASC or DESC)."
)
TOKEN = re.compile(
    r"""\s*(?:
        (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
      | (?P<op>!=|!~|>=|<=|=|~|>|<)
      | (?P<punct>[(),])
      | (?P<word>[^\s=!~<>(),"']+)
    )""",
    re.VERBOSE,
)
FIELDS = {"project", "key", "issuekey", "status", "statuscategory", "assignee", "text"}
ORDER_FIELDS = {"created", "updated", "key", "issuekey", "duedate"}


@dataclass
class Token:
    kind: str  # string, op, punct or word
    text: str

    def word(self, *words: str) -> bool:
        return self.kind == "word" and self.text.upper() in words


@dataclass
class Clause:
    field: str
    op: str  # =, !=, ~, is, is not
    value: str | None  # None: EMPTY


@dataclass
class Query:
    clauses: list[Clause]
    order: list[tuple[str, bool]]  # (field, descending)


def unsupported(detail: str) -> JqlError:
    return JqlError(f"Unsupported JQL: {detail}. {SUPPORTED}")


def tokenize(jql: str) -> list[Token]:
    tokens, at = [], 0
    while jql[at:].strip():
        match = TOKEN.match(jql, at)
        if not match or match.end() == at:
            raise JqlError(f"Invalid JQL: unclosed quote or unexpected character at {at}: {jql!r}")
        kind = match.lastgroup or "word"
        text = match.group(kind)
        if kind == "string":
            text = re.sub(r"\\(.)", r"\1", text[1:-1])
        tokens.append(Token(kind, text))
        at = match.end()
    return tokens


def parse(jql: str) -> Query:
    tokens = tokenize(jql)
    at = 0

    def peek() -> Token | None:
        return tokens[at] if at < len(tokens) else None

    def take(what: str) -> Token:
        nonlocal at
        token = peek()
        if token is None:
            raise JqlError(f"Invalid JQL: expected {what} at the end of {jql!r}")
        at += 1
        return token

    if not tokens or tokens[0].word("ORDER"):
        raise JqlError(
            "Unbounded JQL queries are not allowed here: add at least one search clause, "
            "e.g. project = KEY"
        )
    clauses: list[Clause] = []
    while True:
        clauses.append(clause(take, peek))
        token = peek()
        if token is None:
            return Query(clauses, [])
        if token.word("AND"):
            take("AND")
            continue
        if token.word("ORDER"):
            take("ORDER")
            return Query(clauses, order(take, peek))
        if token.word("OR", "NOT") or token.text in ("(", ")"):
            raise unsupported(f"{token.text!r}")
        raise JqlError(f"Invalid JQL: expected AND or ORDER BY, not {token.text!r}")


def clause(take, peek) -> Clause:
    token = take("a field")
    if token.kind != "word" or token.word("NOT"):
        raise unsupported(f"{token.text!r} where a field name should be")
    field = token.text.lower()
    if field not in FIELDS:
        raise unsupported(f"the field {token.text!r}")
    op_token = take("an operator")
    if op_token.word("IS"):
        negated = bool(peek() and peek().word("NOT"))
        if negated:
            take("NOT")
        if field != "assignee" or not take("EMPTY").word("EMPTY", "NULL"):
            raise unsupported(f"IS on {field}")
        return Clause(field, "is not" if negated else "is", None)
    if op_token.kind != "op" or op_token.text not in ("=", "!=", "~"):
        raise unsupported(f"the operator {op_token.text!r}")
    if (field == "text") != (op_token.text == "~"):
        raise unsupported(f"{field} {op_token.text}")
    value = take("a value")
    following = peek()
    if value.kind not in ("string", "word") or (following and following.text == "("):
        raise unsupported(f"the value {value.text!r} (functions and lists are not supported)")
    if value.kind == "word" and value.word("EMPTY", "NULL"):
        if field != "assignee":
            raise unsupported(f"EMPTY on {field}")
        return Clause(field, "is" if op_token.text == "=" else "is not", None)
    if not value.text.strip():
        raise JqlError(f"Invalid JQL: {field} needs a value")
    return Clause(field, op_token.text, value.text)


def order(take, peek) -> list[tuple[str, bool]]:
    if not take("BY").word("BY"):
        raise JqlError("Invalid JQL: expected BY after ORDER")
    keys = []
    while True:
        token = take("a field to order by")
        if token.kind not in ("word", "string") or token.text.lower() not in ORDER_FIELDS:
            raise unsupported(f"ORDER BY {token.text}")
        descending = False
        if peek() and peek().word("ASC", "DESC"):
            descending = take("ASC or DESC").word("DESC")
        keys.append((token.text.lower(), descending))
        following = peek()
        if following is None:
            return keys
        if following.text != ",":
            raise JqlError(f"Invalid JQL: unexpected {following.text!r} after ORDER BY")
        take(",")


def values(issue: Issue, field: str) -> list[str]:
    """What a clause's value may equal, case-insensitively."""
    fields = issue["fields"]
    if field == "project":
        return [fields["project"]["key"], fields["project"]["name"]]
    if field in ("key", "issuekey"):
        return [issue["key"], issue["id"]]
    if field == "status":
        return [fields["status"]["name"], fields["status"]["id"]]
    if field == "statuscategory":
        category = fields["status"]["statusCategory"]
        return [category["name"], category["key"]]
    if field == "assignee":
        person = fields["assignee"]
        return (
            [person["accountId"], person["displayName"], person["emailAddress"]] if person else []
        )
    raise AssertionError(field)


def check_values(world: World, query: Query) -> None:
    """Jira refuses a value that names nothing, rather than finding no issues."""
    known: dict[str, set[str]] = {
        "project": {v for p in world.projects for v in (p["key"], p["name"])},
        "key": {v for i in world.issues for v in (i["key"], i["id"])},
        "status": {v for s in world.statuses for v in (s["name"], s["id"])},
        "statuscategory": {*STATUS_CATEGORIES, *STATUS_CATEGORIES.values()},
        "assignee": {
            v for a in world.accounts for v in (a["accountId"], a["displayName"], a["emailAddress"])
        },
    }
    known["issuekey"] = known["key"]
    for c in query.clauses:
        if c.value is None or c.field == "text":
            continue
        if c.value.casefold() not in {v.casefold() for v in known[c.field]}:
            raise JqlError(
                f"Invalid JQL: The value '{c.value}' does not exist for the field '{c.field}'."
            )


def matches(issue: Issue, c: Clause) -> bool:
    if c.field == "text":
        fields = issue["fields"]
        haystack = f"{fields['summary']}\n{fields.get('description') or ''}".casefold()
        return all(word in haystack for word in c.value.casefold().split())
    found = [v.casefold() for v in values(issue, c.field)]
    if c.op == "is":
        return not found
    if c.op == "is not":
        return bool(found)
    if c.op == "=":
        return c.value.casefold() in found
    return bool(found) and c.value.casefold() not in found  # !=: never matches an empty field


def sort_value(issue: Issue, field: str) -> Any:
    fields = issue["fields"]
    if field in ("key", "issuekey"):
        project, number = issue["key"].rsplit("-", 1)
        return project, int(number)
    if field == "duedate":
        return fields["duedate"]
    return datetime.fromisoformat(fields[field])


def run(world: World, query: Query) -> list[Issue]:
    check_values(world, query)
    found = [i for i in world.issues if all(matches(i, c) for c in query.clauses)]
    found.sort(key=lambda i: sort_value(i, "key"))
    for field, descending in reversed(query.order):
        present = [i for i in found if sort_value(i, field) is not None]
        present.sort(key=lambda i: sort_value(i, field), reverse=descending)
        found = present + [i for i in found if sort_value(i, field) is None]
    return found


# Tools


@server.tool(annotations=READ)
def searchJiraIssuesUsingJql(
    cloudId: str,
    jql: str,
    fields: list[str] | None = None,
    maxResults: int | None = None,
    nextPageToken: str | None = None,
) -> dict[str, Any]:
    """Issues matching a JQL query: {"issues": [...], "isLast": bool, "nextPageToken"?}. Supports
    clauses joined by AND on project, key, status and statusCategory (= and !=), assignee (=, !=,
    IS [NOT] EMPTY; by account id, name or email) and text ~ "words" (every word, in summary or
    description, ignoring case), then ORDER BY created, updated, key or duedate. Anything else
    is an error, never a wider search. Without ORDER BY, issues come in key order."""
    query = parse(jql)
    size = DEFAULT_RESULTS if maxResults is None else maxResults
    if not 1 <= size <= MAX_RESULTS:
        raise ToolError(f"maxResults must be between 1 and {MAX_RESULTS}")
    try:
        start = int(nextPageToken) if nextPageToken else 0
    except ValueError:
        raise ToolError(f"Invalid nextPageToken {nextPageToken!r}") from None
    found = run(load_world(), query)
    page = [shaped(issue, fields) for issue in found[start : start + size]]
    result: dict[str, Any] = {"issues": page, "isLast": start + size >= len(found)}
    if not result["isLast"]:
        result["nextPageToken"] = str(start + size)
    return result


@server.tool(annotations=READ)
def getJiraIssue(
    cloudId: str, issueIdOrKey: str, fields: list[str] | None = None
) -> dict[str, Any]:
    """One issue by key or id, with every field or only those asked for."""
    return shaped(load_world().find(issueIdOrKey), fields)


@server.tool(annotations=READ)
def getTransitionsForJiraIssue(cloudId: str, issueIdOrKey: str) -> dict[str, Any]:
    """{"transitions": [{"id", "name", "to": status}]}: any status other than the current one."""
    world = load_world()
    return {"transitions": transitions(world, world.find(issueIdOrKey))}


@server.tool(annotations=READ)
def lookupJiraAccountId(cloudId: str, searchString: str) -> dict[str, Any]:
    """Accounts whose name or email matches: {"users": [{"accountId", "displayName",
    "emailAddress"}, ...]}. The brain also accepts a bare list or "result" in place of "users",
    and "account_id"; the real server's shape is unverified. The accounts are the data's
    assignees and reporters; a name or email matches when it contains the search string."""
    needle = searchString.strip().casefold()
    if not needle:
        return {"users": []}
    users = [
        account
        for account in load_world().accounts
        if needle in account["displayName"].casefold()
        or needle in account["emailAddress"].casefold()
        or needle == account["accountId"].casefold()
    ]
    return {"users": users}


@server.tool(annotations=WRITE)
def createJiraIssue(
    cloudId: str,
    projectKey: str,
    issueTypeName: str,
    summary: str,
    description: str | None = None,
    assignee_account_id: str | None = None,
    additional_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Creates the issue under the next key after the project's highest: {"id", "key", "self"}.
    additional_fields may set duedate, priority, labels and parent."""
    arguments = {
        "projectKey": projectKey,
        "issueTypeName": issueTypeName,
        "summary": summary,
        "description": description,
        "assignee_account_id": assignee_account_id,
        "additional_fields": additional_fields,
    }
    issue = write("createJiraIssue", "create_issue", projectKey, arguments)
    return {"id": issue["id"], "key": issue["key"], "self": issue["self"]}


@server.tool(annotations=WRITE)
def editJiraIssue(cloudId: str, issueIdOrKey: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Sets summary, description, assignee ({"accountId"} or null), priority, labels, duedate or
    parent. Status changes go through transitionJiraIssue. Returns the edited issue."""
    return write("editJiraIssue", "edit_issue", issueIdOrKey, {"fields": fields})


@server.tool(annotations=WRITE)
def transitionJiraIssue(
    cloudId: str, issueIdOrKey: str, transition: dict[str, str]
) -> dict[str, Any]:
    """Moves the issue along a transition from getTransitionsForJiraIssue ({"id": ...}) and
    records it in the changelog. Returns the moved issue."""
    arguments = {"transition": transition}
    return write("transitionJiraIssue", "transition_issue", issueIdOrKey, arguments)


@server.tool(annotations=WRITE)
def addCommentToJiraIssue(cloudId: str, issueIdOrKey: str, commentBody: str) -> dict[str, Any]:
    """Adds a plain-text comment; returns it: {"id", "body", "created", "updated"}."""
    arguments = {"commentBody": commentBody}
    issue = write("addCommentToJiraIssue", "add_comment", issueIdOrKey, arguments)
    return issue["fields"]["comment"]["comments"][-1]


def main() -> None:
    server.run("streamable-http", port=Settings().world_jira_mcp_port)
