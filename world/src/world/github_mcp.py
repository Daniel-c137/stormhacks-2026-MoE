"""Mock GitHub MCP server over mock-data/github/ and the snapshot's code, with comments journaled
to the overlay. It serves every repository of world/world.json: the main one (github_repo) and
the extra_github_repos, each with its own issues, pull requests, releases and code.

Tool names, arguments and result shapes copy GitHub's official MCP server (github-mcp-server
v1.14): searches return the REST search result, issue_read, pull_request_read and the list_*
tools its trimmed "minimal" types, as JSON text like the official server. Pointing
GITHUB_MCP_URL at the real server is a config change. The issue, pull request, release and
comment records are the same in every snapshot; the default branch is the snapshot's head
commit (world/snapshots/<name>/repo.json), so list_commits and the code agree. Anything the data
cannot answer is a tool error, never an empty or invented answer. Production is whatever the
latest release contains.
"""

import base64
import json
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import Field

from .code import RepoTree, blob_sha, load_repo, load_tree
from .config import Settings, world_spec
from .overlay import JournalEntry, JournalOverlay
from .snapshot import mock_records

server = MCPServer("github")
READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)
API = "https://api.github.com"

Record = dict[str, Any]
PerPage = Annotated[int, Field(ge=1, le=100)]
Page = Annotated[int, Field(ge=1)]
CodeField = Literal["name", "path", "sha", "repository", "text_matches"]
ContentField = Literal[
    "type", "name", "path", "size", "sha", "url", "git_url", "html_url", "download_url"
]
ListIssueField = Literal[
    "number",
    "title",
    "body",
    "state",
    "user",
    "labels",
    "assignees",
    "comments",
    "created_at",
    "updated_at",
    "field_values",
]
SearchIssueField = Literal[
    "number",
    "title",
    "body",
    "state",
    "state_reason",
    "draft",
    "locked",
    "html_url",
    "user",
    "author_association",
    "labels",
    "assignee",
    "assignees",
    "milestone",
    "comments",
    "reactions",
    "created_at",
    "updated_at",
    "closed_at",
    "closed_by",
    "type",
    "repository_url",
    "pull_request",
    "field_values",
]
SearchPullField = Literal[
    "number",
    "title",
    "body",
    "state",
    "state_reason",
    "draft",
    "locked",
    "html_url",
    "user",
    "author_association",
    "labels",
    "assignee",
    "assignees",
    "milestone",
    "comments",
    "reactions",
    "created_at",
    "updated_at",
    "closed_at",
    "closed_by",
    "pull_request",
    "repository_url",
]
PullField = Literal[
    "number",
    "title",
    "body",
    "state",
    "draft",
    "merged",
    "mergeable_state",
    "html_url",
    "user",
    "labels",
    "assignees",
    "requested_reviewers",
    "merged_by",
    "head",
    "base",
    "additions",
    "deletions",
    "changed_files",
    "commits",
    "comments",
    "created_at",
    "updated_at",
    "closed_at",
    "merged_at",
    "milestone",
]
CommitField = Literal["sha", "html_url", "commit", "author", "committer"]
ReleaseField = Literal[
    "id", "tag_name", "name", "body", "html_url", "published_at", "prerelease", "draft", "author"
]
SearchSort = Literal[
    "comments",
    "reactions",
    "reactions-+1",
    "reactions--1",
    "reactions-smile",
    "reactions-thinking_face",
    "reactions-heart",
    "reactions-tada",
    "interactions",
    "created",
    "updated",
]
_write_lock = threading.Lock()  # a write reads the state, checks it and journals as one step


# The world: the mock data with the journal replayed over it


@dataclass
class World:
    """One repository."""

    full_name: str  # owner/name, from world/world.json
    tree: RepoTree  # the default branch's code at the snapshot's head
    issues: list[Record]
    pulls: list[Record]
    releases: list[Record]
    commits: list[Record]  # newest first, one line of history
    comments: list[Record]
    main: bool = True  # world.json's github_repo; its comments journal as a bare number

    def target(self, number: int) -> str:
        """How the journal names an issue: its number in the main repository, else
        owner/name#number."""
        return str(number) if self.main else f"{self.full_name}#{number}"

    def issue(self, number: int, action: str = "get issue") -> Record:
        """An issue or pull request by number: both are issues to the issues API."""
        for record in (*self.issues, *self.pulls):
            if record["number"] == number:
                return record
        raise ToolError(f"failed to {action}: {self.full_name}#{number}: 404 Not Found")

    def pull(self, number: int) -> Record:
        for record in self.pulls:
            if record["number"] == number:
                return record
        raise ToolError(f"failed to get pull request: {self.full_name}#{number}: 404 Not Found")

    def thread(self, number: int) -> list[Record]:
        return [c for c in self.comments if c["issue_number"] == number]


def load_worlds() -> list[World]:
    """Every repository's mock data with every journaled GitHub write applied, in order."""
    spec = world_spec()
    snapshot = Settings().world_snapshot
    worlds = [
        World(
            full_name=full_name,
            tree=load_repo(snapshot) if main else load_tree(snapshot, "github", full_name),
            issues=mock_records("github", "issues", None if main else full_name),
            pulls=mock_records("github", "pull_requests", None if main else full_name),
            releases=mock_records("github", "releases", None if main else full_name),
            commits=mock_records("github", "commits", None if main else full_name),
            comments=mock_records("github", "comments", None if main else full_name),
            main=main,
        )
        for main, full_name in (
            (True, spec.github_repo),
            *((False, r) for r in spec.extra_github_repos),
        )
    ]
    for entry in JournalOverlay.configured().journal():
        if entry.server == "github":
            repo, _, number = entry.target.rpartition("#")
            world = next((w for w in worlds if w.full_name == repo or (not repo and w.main)), None)
            if world is not None:
                apply(world, entry.model_copy(update={"target": number}))
    return worlds


def load_world(owner: str | None = None, repo: str | None = None, action: str = "get") -> World:
    """One repository: the main one, or owner/repo (ignoring case); 404 for any other."""
    worlds = load_worlds()
    if owner is None and repo is None:
        return worlds[0]
    for world in worlds:
        if f"{owner}/{repo}".casefold() == world.full_name.casefold():
            return world
    raise ToolError(f"failed to {action}: {owner}/{repo}: 404 Not Found")


def apply(world: World, entry: JournalEntry) -> Record:
    """Replays a write; only comments exist. Returns the comment."""
    record = world.issue(int(entry.target), "add comment")
    body = entry.arguments["body"]
    if not body.strip():
        raise ToolError("body cannot be empty when provided")
    comment_id = max((c["id"] for c in world.comments), default=0) + 1
    at = entry.at.strftime("%Y-%m-%dT%H:%M:%SZ")
    comment: Record = {
        "issue_number": record["number"],
        "issue_url": f"{API}/repos/{world.full_name}/issues/{record['number']}",
        "id": comment_id,
        "user": None,  # the mock has no signed-in user
        "body": body,
        "created_at": at,
        "updated_at": at,
        "html_url": f"{record['html_url']}#issuecomment-{comment_id}",
        "author_association": "NONE",
    }
    world.comments.append(comment)
    record["comments"] += 1
    record["updated_at"] = at
    return comment


# Shapes


def is_pull(record: Record) -> bool:
    return "head" in record


def marshalled(payload: Any) -> CallToolResult:
    """JSON text, as the official server returns every result."""
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])


def public(record: Record) -> Record:
    """The record without the data's own bookkeeping keys (_files, _closes, ...)."""
    return {k: v for k, v in record.items() if not k.startswith("_")}


def trimmed(items: list[Record], fields: list[str] | None) -> list[Record]:
    if not fields:
        return items
    return [{k: v for k, v in item.items() if k in fields} for item in items]


def dropped_empty(item: Record, *keys: str) -> Record:
    """Go's omitempty: these keys go when they are empty, zero or false."""
    return {k: v for k, v in item.items() if k not in keys or v not in (None, "", 0, False, [])}


def minimal_user(user: Record | None) -> Record | None:
    if not user:
        return None
    return dropped_empty(
        {"login": user["login"], "id": user.get("id"), "profile_url": user.get("html_url")},
        "id",
        "profile_url",
    )


def names(items: list[Record], key: str) -> list[str]:
    return [item[key] for item in items or []]


def minimal_issue(record: Record) -> Record:
    """An issue (or a pull request read as an issue) as issue_read's get returns it."""
    return dropped_empty(
        {
            "number": record["number"],
            "title": record["title"],
            "body": record.get("body"),
            "state": record["state"],
            "state_reason": record.get("state_reason"),
            "draft": record.get("draft"),
            "locked": record.get("locked"),
            "html_url": record["html_url"],
            "user": minimal_user(record.get("user")),
            "author_association": record.get("author_association"),
            "labels": names(record.get("labels"), "name"),
            "assignees": names(record.get("assignees"), "login"),
            "comments": record.get("comments"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "closed_at": record.get("closed_at"),
        },
        "body",
        "state_reason",
        "draft",
        "locked",
        "user",
        "author_association",
        "labels",
        "comments",
        "created_at",
        "updated_at",
        "closed_at",
    )


def graphql_issue(record: Record) -> Record:
    """An issue as list_issues returns it, from GraphQL: state in capitals, no URL."""
    return dropped_empty(
        {
            "number": record["number"],
            "title": record["title"],
            "body": record.get("body"),
            "state": record["state"].upper(),
            "user": {"login": record["user"]["login"]},
            "labels": names(record.get("labels"), "name"),
            "assignees": names(record.get("assignees"), "login"),
            "comments": record.get("comments"),
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        },
        "body",
        "labels",
        "comments",
    )


def search_item(world: World, record: Record) -> Record:
    """An issue or pull request as the issues search API returns it."""
    repository_url = f"{API}/repos/{world.full_name}"
    if not is_pull(record):
        return {**public(record), "repository_url": repository_url}
    number = record["number"]
    keys = (
        "number title body state user labels assignees comments created_at updated_at "
        "closed_at author_association html_url draft"
    ).split()
    return {
        **{k: record.get(k) for k in keys},
        "url": f"{API}/repos/{world.full_name}/issues/{number}",
        "repository_url": repository_url,
        "state_reason": None,
        "pull_request": {
            "url": record["url"],
            "html_url": record["html_url"],
            "diff_url": f"{record['html_url']}.diff",
            "patch_url": f"{record['html_url']}.patch",
            "merged_at": record.get("merged_at"),
        },
    }


def minimal_branch(branch: Record) -> Record:
    return {"ref": branch["ref"], "sha": branch.get("sha", "")}


def minimal_pull(world: World, record: Record, *, full: bool) -> Record:
    """A pull request as pull_request_read's get returns it (full), or as list_pull_requests
    does: the REST list's simple pull requests carry no merged flag, merge author, size or
    comment count, so there merged is always false and merged_at tells a merge."""
    item: Record = {
        "number": record["number"],
        "title": record["title"],
        "body": record.get("body"),
        "state": record["state"],
        "draft": bool(record.get("draft")),
        "merged": bool(record.get("merged")) if full else False,
        "mergeable_state": record.get("mergeable_state") if full else None,
        "html_url": record["html_url"],
        "user": minimal_user(record.get("user")),
        "labels": names(record.get("labels"), "name"),
        "assignees": names(record.get("assignees"), "login"),
        "requested_reviewers": names(record.get("requested_reviewers"), "login"),
        "merged_by": (record.get("merged_by") or {}).get("login") if full else None,
        "head": minimal_branch(record["head"]),
        "base": minimal_branch(record["base"]),
        "additions": record.get("additions") if full else None,
        "deletions": record.get("deletions") if full else None,
        "changed_files": record.get("changed_files") if full else None,
        "commits": len(pull_commits(world, record["number"])) if full else None,
        "comments": record.get("comments") if full else None,
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "closed_at": record.get("closed_at"),
        "merged_at": record.get("merged_at"),
    }
    return dropped_empty(
        item,
        "body",
        "mergeable_state",
        "user",
        "labels",
        "assignees",
        "requested_reviewers",
        "merged_by",
        "additions",
        "deletions",
        "changed_files",
        "commits",
        "comments",
        "created_at",
        "updated_at",
        "closed_at",
        "merged_at",
    )


def minimal_comment(comment: Record) -> Record:
    return dropped_empty(
        {
            "id": comment["id"],
            "body": comment.get("body"),
            "html_url": comment["html_url"],
            "user": minimal_user(comment.get("user")),
            "author_association": comment.get("author_association"),
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
        },
        "body",
        "user",
        "author_association",
        "created_at",
        "updated_at",
    )


def minimal_commit(commit: Record) -> Record:
    info = commit["commit"]
    item: Record = {
        "sha": commit["sha"],
        "html_url": commit["html_url"],
        "commit": {
            "message": info["message"],
            **{role: dict(info[role]) for role in ("author", "committer") if info.get(role)},
        },
    }
    for role in ("author", "committer"):
        if user := minimal_user(commit.get(role)):
            item[role] = user
    return item


def minimal_release(release: Record) -> Record:
    return dropped_empty(
        {
            "id": release["id"],
            "tag_name": release["tag_name"],
            "name": release.get("name"),
            "body": release.get("body"),
            "html_url": release["html_url"],
            "published_at": release.get("published_at"),
            "prerelease": bool(release.get("prerelease")),
            "draft": bool(release.get("draft")),
            "author": minimal_user(release.get("author")),
        },
        "name",
        "body",
        "published_at",
        "author",
    )


def paged(items: list[Any], page: int | None, per_page: int | None) -> list[Any]:
    size = per_page or 30
    start = ((page or 1) - 1) * size
    return items[start : start + size]


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_iso(value: str, what: str) -> datetime:
    """An RFC 3339 time or a YYYY-MM-DD date (midnight UTC), like the official server."""
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return timestamp(f"{date.fromisoformat(value).isoformat()}T00:00:00Z")
        parsed = timestamp(value)
        if parsed.tzinfo is None:
            raise ValueError(value)
        return parsed
    except ValueError:
        raise ToolError(
            f"invalid {what} timestamp: {value} (supported formats: YYYY-MM-DDThh:mm:ssZ or "
            "YYYY-MM-DD)"
        ) from None


# Search: the official server's prepareSearchArgs, then a subset of GitHub's issue search


SEARCH_SUPPORTED = (
    "This server supports plain words and quoted phrases (every one must appear as whole words "
    "in the title, body or comments, ignoring case) with the qualifiers repo:, org:, user:, "
    "is:issue|pr|open|closed|merged|unmerged|draft, type:issue|pr, state:open|closed, label:, "
    "author: and assignee:."
)
SEARCH_TOKEN = re.compile(r'(-?)(?:([A-Za-z][\w-]*):)?("[^"]*"|[^\s"]+)')
IS_VALUES = {"issue", "pr", "pull-request", "open", "closed", "merged", "unmerged", "draft"}


def has_filter(query: str, name: str, value: str | None = None) -> bool:
    """Whether the query holds name:value (or name:anything), as hasFilter and
    hasSpecificFilter in the official server's search_utils.go decide it."""
    if value is None:
        return bool(re.search(rf"(^|\s|\W){re.escape(name)}:\S+", query))
    return bool(re.search(rf"(^|\s|\W){re.escape(name)}:{re.escape(value)}($|\s|\W)", query))


def prepare_search_args(query: str, kind: str, owner: str | None, repo: str | None) -> str:
    """The query the official server sends: scoped to is:<kind>, and to repo:owner/repo when
    both are given and the query names no repo: of its own."""
    if not query:
        raise ToolError("missing required parameter: query")
    if not has_filter(query, "is", kind):
        query = f"is:{kind} {query}"
    if owner and repo and not has_filter(query, "repo"):
        query = f"repo:{owner}/{repo} {query}"
    return query


@dataclass
class SearchQuery:
    terms: list[str]  # words and phrases, lower case
    qualifiers: list[tuple[str, str]]  # (name, value), lower case


def unsupported_search(detail: str) -> ToolError:
    return ToolError(f"Unsupported search query: {detail}. {SEARCH_SUPPORTED}")


def parse_search(query: str) -> SearchQuery:
    if query.count('"') % 2:
        raise ToolError(f"Invalid search query: unclosed quote in {query!r}")
    terms: list[str] = []
    qualifiers: list[tuple[str, str]] = []
    for negated, name, value in SEARCH_TOKEN.findall(query):
        value = value.strip('"').casefold()
        if negated:
            raise unsupported_search(f"the exclusion -{name + ':' if name else ''}{value}")
        if not name:
            if value.upper() in ("OR", "NOT"):
                raise unsupported_search(f"the operator {value.upper()}")
            if value.upper() != "AND" and value:
                terms.append(value)
            continue
        name = name.casefold()
        if name == "type":
            name = "is"
        if name not in {"repo", "org", "user", "is", "state", "label", "author", "assignee"}:
            raise unsupported_search(f"the qualifier {name}:")
        if (name == "is" and value not in IS_VALUES) or (
            name == "state" and value not in ("open", "closed")
        ):
            raise unsupported_search(f"{name}:{value}")
        qualifiers.append((name, "pr" if value == "pull-request" else value))
    return SearchQuery(terms, qualifiers)


def has_words(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE))


def search_matches(world: World, record: Record, query: SearchQuery) -> bool:
    pull = is_pull(record)
    for name, value in query.qualifiers:
        if name == "repo" or name in ("org", "user"):
            continue  # scope, checked once for the whole search
        if name == "is" and value in ("issue", "pr") and pull != (value == "pr"):
            return False
        if name in ("is", "state") and value in ("open", "closed") and record["state"] != value:
            return False
        if name == "is" and value in ("merged", "unmerged"):
            if not pull or bool(record.get("merged")) != (value == "merged"):
                return False
        if name == "is" and value == "draft" and not record.get("draft"):
            return False
        if name == "label" and value not in {n.casefold() for n in names(record["labels"], "name")}:
            return False
        if name == "author" and record["user"]["login"].casefold() != value:
            return False
        if name == "assignee" and value not in {
            n.casefold() for n in names(record["assignees"], "login")
        }:
            return False
    text = "\n".join(
        [
            record["title"],
            record.get("body") or "",
            *(c["body"] for c in world.thread(record["number"])),
        ]
    )
    return all(has_words(text, term) for term in query.terms)


def in_scope(world: World, query: SearchQuery) -> bool:
    """GitHub ORs repeated repo: (or org:/user:) qualifiers."""
    repos = [v for n, v in query.qualifiers if n == "repo"]
    owners = [v for n, v in query.qualifiers if n in ("org", "user")]
    owner = world.full_name.split("/")[0].casefold()
    return (not repos or world.full_name.casefold() in repos) and (not owners or owner in owners)


def run_search(
    worlds: list[World],
    kind: Literal["issue", "pr"],
    query: str,
    owner: str | None,
    repo: str | None,
    sort: str | None,
    order: str | None,
    page: int | None,
    per_page: int | None,
    fields: list[str] | None,
) -> CallToolResult:
    if sort not in (None, "created", "updated", "comments"):
        raise ToolError(f"sort {sort!r} is not supported here: use created, updated or comments")
    parsed = parse_search(prepare_search_args(query.strip(), kind, owner, repo))
    found: list[tuple[World, Record]] = [
        (world, r)
        for world in worlds
        if in_scope(world, parsed)
        for r in (*world.issues, *world.pulls)
        if search_matches(world, r, parsed)
    ]
    key = {"created": "created_at", "comments": "comments"}.get(sort or "", "updated_at")
    found.sort(key=lambda wr: wr[1]["number"], reverse=True)
    found.sort(key=lambda wr: wr[1][key], reverse=sort is None or order != "asc")
    items = [search_item(world, r) for world, r in paged(found, page, per_page)]
    return marshalled(
        {"total_count": len(found), "incomplete_results": False, "items": trimmed(items, fields)}
    )


# Code


def repo_trees() -> list[tuple[str, RepoTree]]:
    """Every repository (owner/name) with its files at the configured snapshot."""
    snapshot = Settings().world_snapshot
    spec = world_spec()
    return [
        (spec.github_repo, load_repo(snapshot)),
        *((r, load_tree(snapshot, "github", r)) for r in spec.extra_github_repos),
    ]


CODE_QUALIFIERS = {"repo", "org", "user", "path", "filename", "extension"}


@server.tool(annotations=READ)
def search_code(
    query: str,
    sort: str | None = None,
    order: Literal["asc", "desc"] | None = None,
    page: int | None = None,
    perPage: int | None = None,
    fields: list[CodeField] | None = None,
) -> dict[str, Any]:
    """Code search over the default branch, in the official server's minimal result shape.
    Supports plain words (all must match the file's text or path, case-insensitively) and the
    repo:, org:, user:, path:, filename: and extension: qualifiers; other qualifiers, OR, NOT
    and exclusions are an error."""
    words: list[str] = []
    paths: list[str] = []
    repos: list[str] = []
    owners: list[str] = []
    for token in query.split():
        key, colon, value = token.partition(":")
        key, value = key.casefold(), value.strip('"')
        if token.startswith("-") or token in ("OR", "NOT"):
            raise ToolError(f"Unsupported code search syntax {token!r}: use plain words")
        if not colon:
            if token != "AND":
                words.append(token.strip('"').casefold())
        elif key not in CODE_QUALIFIERS:
            raise ToolError(
                f"Unsupported code search qualifier {key}: (supported: "
                f"{', '.join(sorted(CODE_QUALIFIERS))})"
            )
        elif key == "repo":
            repos.append(value.casefold())  # repeated repo: (or org:/user:) are ORed
        elif key in ("org", "user"):
            owners.append(value.casefold())
        elif key == "path":
            paths.append(value.strip("/"))
        elif key == "filename":
            paths.append(f"*{value}")
        elif key == "extension":
            paths.append(f"*.{value.lstrip('.')}")

    def path_ok(path: str) -> bool:
        return all(path.endswith(p[1:]) if p.startswith("*") else path.startswith(p) for p in paths)

    items = [
        {
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "sha": blob_sha(text),
            "repository": full_name,
        }
        for full_name, tree in repo_trees()
        if (not repos or full_name.casefold() in repos)
        and (not owners or full_name.split("/")[0].casefold() in owners)
        for path, text in tree.files.items()
        if path_ok(path) and all(w in path.casefold() or w in text.casefold() for w in words)
    ]
    size = perPage or 30
    start = ((page or 1) - 1) * size
    shown = items[start : start + size]
    if fields:
        shown = [{k: v for k, v in item.items() if k in fields} for item in shown]
    return {"total_count": len(items), "incomplete_results": False, "items": shown}


@server.tool(annotations=READ)
def get_file_contents(
    owner: str,
    repo: str,
    path: str = "/",
    ref: str | None = None,
    sha: str | None = None,
    fields: list[ContentField] | None = None,
) -> CallToolResult:
    """A file as a text message plus an embedded resource whose URI names the commit, or a
    directory as a JSON list of entries, like the official server. Only the default branch's
    head commit exists."""
    found = [(n, t) for n, t in repo_trees() if f"{owner}/{repo}".casefold() == n.casefold()]
    if not found:
        raise ToolError(f"failed to get repository info: {owner}/{repo}: 404 Not Found")
    ((full_name, tree),) = found
    branch = (ref or tree.branch).removeprefix("refs/heads/")
    if (sha and sha != tree.sha) or branch != tree.branch:
        raise ToolError(f"failed to resolve git reference {sha or ref}: 404 Not Found")
    clean = path.strip().strip("/")
    if ".." in clean.split("/") or "\\" in clean:
        raise ToolError(f"invalid path {path!r}")

    if clean in tree.files:
        text = tree.files[clean]
        return CallToolResult(
            content=[
                TextContent(
                    type="text", text=f"successfully downloaded text file (SHA: {blob_sha(text)})"
                ),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"repo://{full_name}/sha/{tree.sha}/contents/{clean}",
                        mime_type="text/plain; charset=utf-8",
                        text=text,
                    ),
                ),
            ]
        )

    prefix = f"{clean}/" if clean else ""
    entries: dict[str, dict[str, Any]] = {}
    for file_path, text in tree.files.items():
        if not file_path.startswith(prefix):
            continue
        name, _, rest = file_path[len(prefix) :].partition("/")
        child = f"{prefix}{name}"
        entries.setdefault(
            child,
            {
                "type": "dir" if rest else "file",
                "name": name,
                "path": child,
                "size": 0 if rest else len(text.encode()),
                "sha": "" if rest else blob_sha(text),
                "html_url": f"https://github.com/{full_name}/{'tree' if rest else 'blob'}/"
                f"{tree.sha}/{child}",
            },
        )
    if not entries:
        raise ToolError(f"failed to get file contents {path}: 404 Not Found")
    listing = list(entries.values())
    if fields:
        listing = [{k: v for k, v in entry.items() if k in fields} for entry in listing]
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(listing))])


# Issues


def cursor(offset: int) -> str:
    return base64.b64encode(f"cursor:{offset}".encode()).decode()


def offset_of(after: str) -> int:
    """The index after the item a cursor names; cursors number items from 1."""
    try:
        kind, _, offset = base64.b64decode(after, validate=True).decode().partition(":")
        if kind != "cursor" or int(offset) < 0:
            raise ValueError(after)
        return int(offset)
    except ValueError:
        raise ToolError(f"invalid cursor {after!r}: use the endCursor of a previous page") from None


@server.tool(annotations=READ)
def list_issues(
    owner: str,
    repo: str,
    state: Literal["OPEN", "CLOSED"] | None = None,
    labels: list[str] | None = None,
    orderBy: Literal["CREATED_AT", "UPDATED_AT", "COMMENTS"] | None = None,
    direction: Literal["ASC", "DESC"] | None = None,
    since: str | None = None,
    perPage: PerPage | None = None,
    after: str | None = None,
    page: int | None = None,
    field_filters: list[dict[str, str]] | None = None,
    fields: list[ListIssueField] | None = None,
) -> CallToolResult:
    """The repository's issues (not pull requests), by default every one, newest first:
    {"issues", "totalCount", "pageInfo"}. labels matches an issue with any of them; since keeps
    issues updated at or after it. Pages by cursor (after), not page."""
    world = load_world(owner, repo, "list issues")
    if page is not None:
        raise ToolError(
            "This tool uses cursor-based pagination. Use the 'after' parameter with the "
            "'endCursor' value from the previous response instead of 'page'."
        )
    if field_filters:
        raise ToolError("failed to list issues: this repository has no custom issue fields")
    found = [i for i in world.issues if not state or i["state"].upper() == state]
    if labels:
        wanted = {label.casefold() for label in labels}
        found = [i for i in found if wanted & {n.casefold() for n in names(i["labels"], "name")}]
    if since:
        start = parse_iso(since, "since")
        found = [i for i in found if timestamp(i["updated_at"]) >= start]
    key = {"UPDATED_AT": "updated_at", "COMMENTS": "comments"}.get(orderBy or "", "created_at")
    found.sort(key=lambda i: i[key], reverse=direction != "ASC")

    first = offset_of(after) if after else 0
    size = perPage or 30
    shown = found[first : first + size]
    page_info: Record = {
        "hasNextPage": first + size < len(found),
        "hasPreviousPage": first > 0,
    }
    if shown:
        page_info |= {"startCursor": cursor(first + 1), "endCursor": cursor(first + len(shown))}
    issues = trimmed([graphql_issue(i) for i in shown], fields)
    return marshalled({"issues": issues, "totalCount": len(found), "pageInfo": page_info})


@server.tool(annotations=READ)
def search_issues(
    query: str,
    owner: str | None = None,
    repo: str | None = None,
    sort: SearchSort | None = None,
    order: Literal["asc", "desc"] | None = None,
    page: Page | None = None,
    perPage: PerPage | None = None,
    fields: list[SearchIssueField] | None = None,
) -> CallToolResult:
    """Issues matching the query, scoped like the official server: is:issue is added, and
    repo:owner/repo when owner and repo are given and the query has no repo: of its own.
    Plain words and quoted phrases must each appear as whole words in the title, body or
    comments, ignoring case. Qualifiers: repo:, org:, user:, is:/type:, state:, label:, author:,
    assignee:; anything else is an error, never a wider search. Best match lists the most
    recently updated first."""
    return run_search(
        load_worlds(), "issue", query, owner, repo, sort, order, page, perPage, fields
    )


@server.tool(annotations=READ)
def issue_read(
    method: Literal["get", "get_comments", "get_sub_issues", "get_parent", "get_labels"],
    owner: str,
    repo: str,
    issue_number: int,
    page: Page | None = None,
    perPage: PerPage | None = None,
) -> CallToolResult:
    """One issue (or pull request, as an issue): get, with the pull requests set to close it;
    get_comments; get_labels. The data has no issue hierarchy, so get_sub_issues and
    get_parent are errors."""
    world = load_world(owner, repo, "get issue")
    record = world.issue(issue_number)
    if method == "get":
        issue = minimal_issue(record)
        if not is_pull(record):
            closing = [p for p in world.pulls if issue_number in p.get("_closes", [])]
            issue["closed_by_pull_requests"] = {
                "total_count": len(closing),
                "references": [
                    {
                        "number": p["number"],
                        "title": p["title"],
                        "state": "MERGED" if p.get("merged") else p["state"].upper(),
                        "url": p["html_url"],
                        "repository": world.full_name,
                    }
                    for p in closing
                ],
            }
        return marshalled(issue)
    if method == "get_comments":
        thread = paged(world.thread(issue_number), page, perPage)
        return marshalled([minimal_comment(c) for c in thread])
    if method == "get_labels":
        labels = [
            {"id": "", "name": n, "color": "", "description": ""}
            for n in names(record["labels"], "name")
        ]
        return marshalled({"labels": labels, "totalCount": len(labels)})
    raise ToolError(f"{method}: the demo world has no sub-issue or parent data")


# Pull requests


@server.tool(annotations=READ)
def list_pull_requests(
    owner: str,
    repo: str,
    state: Literal["open", "closed", "all"] | None = None,
    head: str | None = None,
    base: str | None = None,
    sort: Literal["created", "updated", "popularity", "long-running"] | None = None,
    direction: Literal["asc", "desc"] | None = None,
    page: Page | None = None,
    perPage: PerPage | None = None,
    fields: list[PullField] | None = None,
) -> CallToolResult:
    """The repository's pull requests (open by default), newest first, like the REST list:
    these carry merged_at but no merged flag (always false here), merge author or size; read
    one with pull_request_read for those. head is owner:branch or a branch; popularity sorts by
    comments; long-running is not supported."""
    world = load_world(owner, repo, "list pull requests")
    if sort == "long-running":
        raise ToolError("sort 'long-running' is not supported here: use created or updated")
    wanted = state or "open"
    found = [p for p in world.pulls if wanted == "all" or p["state"] == wanted]
    if head:
        found = [
            p
            for p in found
            if head.casefold() in (p["head"]["label"].casefold(), p["head"]["ref"].casefold())
        ]
    if base:
        found = [p for p in found if p["base"]["ref"] == base]
    key = {"updated": "updated_at", "popularity": "comments"}.get(sort or "", "created_at")
    descending = direction == "desc" if direction else sort in (None, "created")
    found.sort(key=lambda p: p[key], reverse=descending)
    pulls = [minimal_pull(world, p, full=False) for p in paged(found, page, perPage)]
    return marshalled(trimmed(pulls, fields))


@server.tool(annotations=READ)
def search_pull_requests(
    query: str,
    owner: str | None = None,
    repo: str | None = None,
    sort: SearchSort | None = None,
    order: Literal["asc", "desc"] | None = None,
    page: Page | None = None,
    perPage: PerPage | None = None,
    fields: list[SearchPullField] | None = None,
) -> CallToolResult:
    """Pull requests matching the query, as search_issues but scoped to is:pr; items are issue
    search results whose pull_request holds merged_at. Also supports is:merged, is:unmerged and
    is:draft."""
    return run_search(load_worlds(), "pr", query, owner, repo, sort, order, page, perPage, fields)


def pull_commits(world: World, number: int) -> list[Record]:
    return [c for c in world.commits if c.get("_pr_number") == number]


@server.tool(annotations=READ)
def pull_request_read(
    method: Literal[
        "get",
        "get_diff",
        "get_status",
        "get_files",
        "get_commits",
        "get_review_comments",
        "get_reviews",
        "get_comments",
        "get_check_runs",
    ],
    owner: str,
    repo: str,
    pullNumber: int,
    page: Page | None = None,
    perPage: PerPage | None = None,
    after: str | None = None,
) -> CallToolResult:
    """One pull request: get (with merged, merged_at and merged_by), get_status (the combined
    commit status), get_files, get_commits (its merge commit) and get_comments. The data has
    no diffs, reviews or check runs, so those methods are errors."""
    world = load_world(owner, repo, "get pull request")
    record = world.pull(pullNumber)
    if method == "get":
        return marshalled(minimal_pull(world, record, full=True))
    if method == "get_status":
        status = record.get("_status") or {}
        statuses = status.get("statuses", [])
        return marshalled(
            {
                "state": status.get("state", "pending"),
                "sha": record["head"]["sha"],
                "total_count": len(statuses),
                "statuses": [{"state": s["state"], "context": s["context"]} for s in statuses],
            }
        )
    if method == "get_files":
        files = paged(record.get("_files", []), page, perPage)
        return marshalled(
            [
                dropped_empty(
                    {
                        k: f.get(k)
                        for k in ("filename", "status", "additions", "deletions", "changes")
                    },
                    "status",
                    "additions",
                    "deletions",
                    "changes",
                )
                for f in files
            ]
        )
    if method == "get_commits":
        return marshalled(
            [
                {
                    "sha": c["sha"],
                    "html_url": c["html_url"],
                    "message": c["commit"]["message"],
                    "author": dict(c["commit"]["author"]),
                }
                for c in paged(pull_commits(world, pullNumber), page, perPage)
            ]
        )
    if method == "get_comments":
        thread = paged(world.thread(pullNumber), page, perPage)
        return marshalled([minimal_comment(c) for c in thread])
    missing = {
        "get_diff": "diffs",
        "get_reviews": "reviews",
        "get_review_comments": "review comments",
        "get_check_runs": "check runs (only commit statuses: use get_status)",
    }[method]
    raise ToolError(f"{method}: the demo world has no {missing}")


# Commits and releases


def resolve(world: World, ref: str | None) -> int:
    """The index in world.commits of the commit a branch, tag or SHA names. The default branch
    is the snapshot's head."""
    wanted = (ref or world.tree.branch).strip()
    if wanted.removeprefix("refs/heads/") == world.tree.branch:
        wanted = world.tree.sha
    tag = wanted.removeprefix("refs/tags/")
    for release in world.releases:
        if release["tag_name"] == tag:
            wanted = release["target_commitish"]
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", wanted):
        found = [i for i, c in enumerate(world.commits) if c["sha"].startswith(wanted.lower())]
        if len(found) == 1:
            return found[0]
    raise ToolError(f"failed to list commits: {ref}: 404 No commit found for SHA: {ref}")


def touches(world: World, commit: Record, path: str) -> bool:
    """Whether the commit's pull request changed the path (a file, or a directory's files)."""
    number = commit.get("_pr_number")
    pull = next((p for p in world.pulls if p["number"] == number), None)
    clean = path.strip("/")
    files = [f["filename"] for f in (pull or {}).get("_files", [])]
    return any(f == clean or f.startswith(f"{clean}/") for f in files)


@server.tool(annotations=READ)
def list_commits(
    owner: str,
    repo: str,
    sha: str | None = None,
    author: str | None = None,
    path: str | None = None,
    since: str | None = None,
    until: str | None = None,
    page: Page | None = None,
    perPage: PerPage | None = None,
    fields: list[CommitField] | None = None,
) -> CallToolResult:
    """Commits reachable from sha (a branch, tag or commit SHA; default: the default branch),
    newest first. author is a login or an email; path keeps commits whose pull request changed
    it; since and until bound the commit date."""
    world = load_world(owner, repo, "list commits")
    found = world.commits[resolve(world, sha) :]
    if author:
        who = author.casefold()
        found = [
            c
            for c in found
            if who
            in (
                (c.get("author") or {}).get("login", "").casefold(),
                c["commit"]["author"]["email"].casefold(),
            )
        ]
    if path:
        found = [c for c in found if touches(world, c, path)]
    if since:
        start = parse_iso(since, "since")
        found = [c for c in found if timestamp(c["commit"]["committer"]["date"]) >= start]
    if until:
        end = parse_iso(until, "until")
        found = [c for c in found if timestamp(c["commit"]["committer"]["date"]) <= end]
    commits = [minimal_commit(c) for c in paged(found, page, perPage)]
    return marshalled(trimmed(commits, fields))


def newest_first(releases: list[Record]) -> list[Record]:
    return sorted(releases, key=lambda r: r["created_at"], reverse=True)


@server.tool(annotations=READ)
def list_releases(
    owner: str,
    repo: str,
    page: Page | None = None,
    perPage: PerPage | None = None,
    fields: list[ReleaseField] | None = None,
) -> CallToolResult:
    """The repository's releases, newest first."""
    world = load_world(owner, repo, "list releases")
    releases = [minimal_release(r) for r in paged(newest_first(world.releases), page, perPage)]
    return marshalled(trimmed(releases, fields))


@server.tool(annotations=READ)
def get_latest_release(owner: str, repo: str) -> CallToolResult:
    """The newest release that is neither a draft nor a prerelease, as the REST API returns it.
    Whatever it contains is in production; anything merged later is not."""
    world = load_world(owner, repo, "get latest release")
    full = [r for r in newest_first(world.releases) if not r["draft"] and not r["prerelease"]]
    if not full:
        raise ToolError(f"failed to get latest release: {world.full_name}: 404 Not Found")
    return marshalled(public(full[0]))


# Writes


@server.tool(annotations=WRITE)
def add_issue_comment(owner: str, repo: str, issue_number: int, body: str) -> CallToolResult:
    """Comments on an issue or pull request: {"id", "url"}. Writes go to the overlay journal,
    never to the mock data."""
    with _write_lock:
        world = load_world(owner, repo, "add comment")
        target = world.target(world.issue(issue_number, "add comment")["number"])
        entry = JournalEntry.now(
            "github", "add_issue_comment", "add_comment", target, {"body": body}
        )
        comment = apply(world, entry.model_copy(update={"target": str(issue_number)}))
        JournalOverlay.configured().record(entry)
    return marshalled({"id": str(comment["id"]), "url": comment["html_url"]})


def main() -> None:
    settings = Settings()
    server.run("streamable-http", host=settings.world_mcp_host, port=settings.world_github_mcp_port)
