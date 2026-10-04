"""Mock GitLab MCP server over mock-data/gitlab/ and the snapshot's code. Read-only.

Tool names and arguments copy GitLab's official MCP server as documented for GitLab 19.5
(docs.gitlab.com/user/model_context_protocol/mcp_server_tools): get_mcp_server_version,
get_project, search, get_work_item, list_work_items, get_merge_request, list_merge_requests,
get_repository_file, list_repository_tree and list_releases. The documentation names arguments
but not every result's exact shape, so results use GitLab's REST shapes (search, merge requests,
files) and its GraphQL shape for work items, as JSON text. Pointing GITLAB_MCP_URL at a real
GitLab (https://<instance>/api/v4/mcp) is a config change.

It serves every project in world.json's gitlab_projects; the records are the same in every
snapshot, and the default branch is the snapshot's head (world/snapshots/<name>/repos/gitlab/).
Anything the data cannot answer is a tool error, never an empty or invented answer.
"""

import base64
import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlparse

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from .code import RepoTree, load_tree
from .config import Settings, world_spec
from .snapshot import mock_records

server = MCPServer("gitlab")
READ = ToolAnnotations(read_only_hint=True)
VERSION = "19.5.0-world"

Record = dict[str, Any]
First = Annotated[int, Field(ge=1, le=100)]
SEARCH_SCOPES = (
    "work_items",
    "merge_requests",
    "projects",
    "blobs",
    "commits",
    "milestones",
    "users",
    "issues",
    "snippets",
    "wiki_pages",
)
SUPPORTED_SCOPES = ("work_items", "issues", "merge_requests", "blobs", "projects")
MR_FACETS = ("diffs", "commits", "notes", "pipelines", "discussions", "approvals", "conflicts")
MAX_FILE_LINES = 2000


@dataclass
class Project:
    meta: Record  # id, path_with_namespace, default_branch, web_url, ...
    tree: RepoTree
    issues: list[Record]
    merge_requests: list[Record]
    notes: list[Record]
    releases: list[Record]

    @property
    def path(self) -> str:
        return self.meta["path_with_namespace"]

    def issue(self, iid: int) -> Record:
        for record in self.issues:
            if record["iid"] == iid:
                return record
        raise ToolError(f"404 Work item {self.path}#{iid} Not Found")

    def merge_request(self, iid: int) -> Record:
        for record in self.merge_requests:
            if record["iid"] == iid:
                return record
        raise ToolError(f"404 Merge request {self.path}!{iid} Not Found")


def load_projects() -> list[Project]:
    snapshot = Settings().world_snapshot
    return [
        Project(
            meta=mock_records("gitlab", "project", path),
            tree=load_tree(snapshot, "gitlab", path),
            issues=mock_records("gitlab", "issues", path),
            merge_requests=mock_records("gitlab", "merge_requests", path),
            notes=mock_records("gitlab", "notes", path),
            releases=mock_records("gitlab", "releases", path),
        )
        for path in world_spec().gitlab_projects
    ]


def find_project(project_id: str | None = None, url: str | None = None) -> Project:
    """By numeric id or full path (ignoring case), or by any web URL inside the project."""
    if url:
        path = unquote(urlparse(url).path).strip("/").split("/-/", 1)[0]
        project_id = project_id or path
    if not project_id:
        raise ToolError("Provide exactly one of url or project_id")
    wanted = str(project_id).strip().strip("/").casefold()
    for project in load_projects():
        if wanted in (str(project.meta["id"]), project.path.casefold()):
            return project
    raise ToolError(f"404 Project Not Found: {project_id}")


def in_url(url: str | None, kind: str) -> int | None:
    """The iid in a merge request or work item URL, e.g. .../-/merge_requests/4."""
    if url and (m := re.search(rf"/-/{kind}/(\d+)", url)):
        return int(m.group(1))
    return None


def marshalled(payload: Any) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])


def public(record: Record) -> Record:
    """Without the data's own bookkeeping keys (_diffs, _closes)."""
    return {k: v for k, v in record.items() if not k.startswith("_")}


def words(text: str) -> list[str]:
    return [w.casefold() for w in text.split() if w.strip()]


def matches(text: str, terms: list[str]) -> bool:
    low = text.casefold()
    return all(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", low) for t in terms)


def paged(items: list[Any], page: int | None, per_page: int | None) -> list[Any]:
    size = per_page or 20
    start = ((page or 1) - 1) * size
    return items[start : start + size]


# Shapes


def user(raw: Record | None) -> Record | None:
    return {"username": raw["username"], "name": raw["name"]} if raw else None


def work_item(project: Project, record: Record) -> Record:
    """An issue as a GraphQL work item."""
    return {
        "id": f"gid://gitlab/WorkItem/{record['id']}",
        "iid": str(record["iid"]),
        "title": record["title"],
        "state": "OPEN" if record["state"] == "opened" else "CLOSED",
        "description": record.get("description"),
        "webUrl": record["web_url"],
        "reference": f"{project.path}#{record['iid']}",
        "workItemType": {"name": "Issue"},
        "author": user(record.get("author")),
        "assignees": [user(a) for a in record.get("assignees", [])],
        "labels": [{"title": label} for label in record.get("labels", [])],
        "createdAt": record["created_at"],
        "updatedAt": record["updated_at"],
        "closedAt": record.get("closed_at"),
    }


def compact_work_item(project: Project, record: Record) -> Record:
    item = work_item(project, record)
    keys = ("id", "iid", "title", "state", "webUrl", "reference", "createdAt", "updatedAt")
    return {**{k: item[k] for k in keys}, "workItemType": item["workItemType"]}


def compact_merge_request(project: Project, record: Record) -> Record:
    return {
        "iid": record["iid"],
        "title": record["title"],
        "state": record["state"],
        "draft": record.get("draft", False),
        "web_url": record["web_url"],
        "reference": f"{project.path}!{record['iid']}",
        "author": user(record.get("author")),
        "source_branch": record["source_branch"],
        "target_branch": record["target_branch"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "merged_at": record.get("merged_at"),
    }


def blob_hit(project: Project, path: str, text: str, terms: list[str]) -> Record:
    """A code search hit as GitLab's blobs scope returns it: the lines around the first match."""
    lines = text.split("\n")
    first = next((i for i, line in enumerate(lines) if any(t in line.casefold() for t in terms)), 0)
    start = max(first - 2, 0)
    return {
        "basename": path.rsplit(".", 1)[0] if "." in path.rsplit("/", 1)[-1] else path,
        "data": "\n".join(lines[start : start + 6]),
        "path": path,
        "filename": path,
        "id": None,
        "ref": project.tree.branch,
        "startline": start + 1,
        "project_id": project.meta["id"],
    }


# Tools


@server.tool(annotations=READ)
def get_mcp_server_version() -> str:
    """The server's version."""
    return VERSION


@server.tool(annotations=READ)
def get_project(url: str | None = None, project_id: str | None = None) -> CallToolResult:
    """A project's metadata: numeric id, full path, default branch, visibility and web URL."""
    return marshalled(find_project(project_id, url).meta)


@server.tool(annotations=READ)
def search(
    scope: str,
    search: str,
    group_id: str | None = None,
    project_id: str | None = None,
    state: str | None = None,
    confidential: bool | None = None,
    fields: list[str] | None = None,
    order_by: Literal["created_at", "updated_at"] | None = None,
    sort: Literal["asc", "desc"] | None = None,
    per_page: Annotated[int, Field(ge=1, le=100)] | None = None,
    page: Annotated[int, Field(ge=1)] | None = None,
) -> CallToolResult:
    """Searches issues (work_items or issues), merge_requests, blobs (code) or projects, across
    the instance, a group or one project. Every word must appear as a whole word (code: anywhere
    in the file or its path), ignoring case. Other scopes are an error here."""
    if scope not in SEARCH_SCOPES:
        raise ToolError(f"scope does not have a valid value: {scope}")
    if scope not in SUPPORTED_SCOPES:
        raise ToolError(f"The demo world has no {scope}; supported: {', '.join(SUPPORTED_SCOPES)}")
    terms = words(search)
    if not terms:
        raise ToolError("search is missing")
    if project_id:
        projects = [find_project(project_id)]
    else:
        group = (group_id or "").strip("/").casefold()
        projects = [
            p for p in load_projects() if not group or p.path.casefold().startswith(f"{group}/")
        ]

    if scope == "projects":
        found = [
            p.meta for p in projects if matches(f"{p.path} {p.meta.get('description')}", terms)
        ]
        return marshalled(paged(found, page, per_page))
    if scope == "blobs":
        hits = [
            blob_hit(p, path, text, terms)
            for p in projects
            for path, text in p.tree.files.items()
            if all(t in path.casefold() or t in text.casefold() for t in terms)
        ]
        return marshalled(paged(hits, page, per_page))

    searched = set(fields or ["title", "description"])
    if not searched <= {"title", "description"}:
        raise ToolError("fields may only hold title and description")
    if confidential:
        return marshalled([])  # the demo world has no confidential items
    records: list[tuple[Project, Record]] = [
        (p, r)
        for p in projects
        for r in (p.merge_requests if scope == "merge_requests" else p.issues)
        if (not state or state == "all" or r["state"] == state)
        and matches(" ".join(str(r.get(f) or "") for f in sorted(searched)), terms)
    ]
    key = order_by or "created_at"
    records.sort(key=lambda pr: pr[1][key], reverse=sort != "asc")
    return marshalled([public(r) for _, r in paged(records, page, per_page)])


@server.tool(annotations=READ)
def get_work_item(
    url: str | None = None,
    group_id: str | None = None,
    project_id: str | None = None,
    work_item_iid: int | None = None,
    include: list[Literal["notes", "related_merge_requests"]] | None = None,
    notes_first: First | None = None,
    notes_after: str | None = None,
    notes_last: First | None = None,
    notes_before: str | None = None,
    related_merge_requests_first: First | None = None,
    related_merge_requests_after: str | None = None,
) -> CallToolResult:
    """One work item (here, issues) with its type, dates, assignees and labels; include one
    facet: notes or related_merge_requests."""
    if group_id and not project_id:
        raise ToolError("The demo world has no group-level work items")
    project = find_project(project_id, url)
    iid = work_item_iid or in_url(url, "issues") or in_url(url, "work_items")
    if iid is None:
        raise ToolError("Provide work_item_iid with project_id, or a work item url")
    record = project.issue(iid)
    item = work_item(project, record)
    if include and len(include) > 1:
        raise ToolError("include accepts one facet per call")
    if include == ["notes"]:
        notes = [n for n in project.notes if n["noteable_iid"] == iid]
        item["notes"] = [
            {
                "id": f"gid://gitlab/Note/{n['id']}",
                "body": n["body"],
                "author": user(n["author"]),
                "createdAt": n["created_at"],
                "system": n["system"],
            }
            for n in notes[: notes_first or 100]
        ]
    elif include == ["related_merge_requests"]:
        related = [m for m in project.merge_requests if iid in m.get("_closes", [])]
        item["relatedMergeRequests"] = [
            compact_merge_request(project, m) for m in related[: related_merge_requests_first or 20]
        ]
    return marshalled(item)


@server.tool(annotations=READ)
def list_work_items(
    url: str | None = None,
    group_id: str | None = None,
    project_id: str | None = None,
    state: Literal["opened", "closed", "all"] | None = None,
    search: str | None = None,
    author_username: str | None = None,
    assignee_usernames: list[str] | None = None,
    label_name: list[str] | None = None,
    types: list[str] | None = None,
    sort: str | None = None,
    first: First | None = None,
    after: str | None = None,
) -> CallToolResult:
    """Work items (here, issues) of a project: ID, IID, title, state, web URL, reference,
    created and updated times and type, newest first, with cursor pagination."""
    if group_id and not project_id and not url:
        projects = [
            p for p in load_projects() if p.path.casefold().startswith(f"{group_id.casefold()}/")
        ]
    else:
        projects = [find_project(project_id, url)]
    if types and not {t.upper() for t in types} & {"ISSUE"}:
        return marshalled({"items": [], "pageInfo": {"hasNextPage": False, "endCursor": None}})
    terms = words(search or "")
    found = [
        (p, r)
        for p in projects
        for r in p.issues
        if (not state or state == "all" or r["state"] == state)
        and (not terms or matches(f"{r['title']} {r.get('description') or ''}", terms))
        and (not author_username or r["author"]["username"] == author_username)
        and (
            not assignee_usernames
            or {a["username"] for a in r["assignees"]} & set(assignee_usernames)
        )
        and (not label_name or set(r["labels"]) & set(label_name))
    ]
    key = "updated_at" if (sort or "").upper().startswith("UPDATED") else "created_at"
    found.sort(key=lambda pr: pr[1][key], reverse=not (sort or "").upper().endswith("_ASC"))
    start = cursor_offset(after)
    size = first or 20
    shown = found[start : start + size]
    return marshalled(
        {
            "items": [compact_work_item(p, r) for p, r in shown],
            "pageInfo": {
                "hasNextPage": start + size < len(found),
                "endCursor": cursor(start + len(shown)) if shown else None,
            },
        }
    )


def cursor(offset: int) -> str:
    return base64.b64encode(f"offset:{offset}".encode()).decode()


def cursor_offset(after: str | None) -> int:
    if not after:
        return 0
    try:
        kind, _, offset = base64.b64decode(after, validate=True).decode().partition(":")
        if kind != "offset" or int(offset) < 0:
            raise ValueError(after)
        return int(offset)
    except ValueError:
        raise ToolError(f"invalid cursor {after!r}: use endCursor from the previous page") from None


@server.tool(annotations=READ)
def get_merge_request(
    url: str | None = None,
    project_id: str | None = None,
    merge_request_iid: int | None = None,
    include: list[Literal[MR_FACETS]] | None = None,  # type: ignore[valid-type]
    detail: Literal["none", "stats", "full_patch"] | None = None,
    diffs_after: str | None = None,
    diffs_first: First | None = None,
    notes_after: str | None = None,
    notes_first: First | None = None,
    notes_before: str | None = None,
    notes_last: First | None = None,
    commits_after: str | None = None,
    commits_first: First | None = None,
    pipelines_after: str | None = None,
    pipelines_first: First | None = None,
) -> CallToolResult:
    """A merge request; include one facet: diffs (detail none, stats or full_patch; default
    stats), commits or notes. The data has no pipelines, discussions, approvals or conflicts."""
    project = find_project(project_id, url)
    iid = merge_request_iid or in_url(url, "merge_requests")
    if iid is None:
        raise ToolError("Provide merge_request_iid with project_id, or a merge request url")
    record = project.merge_request(iid)
    result = public(record)
    if include and len(include) > 1:
        raise ToolError("include accepts one facet per call")
    facet = include[0] if include else None
    if facet == "diffs":
        level = detail or "stats"
        start = cursor_offset(diffs_after)
        size = diffs_first or 20
        diffs = record["_diffs"][start : start + size]
        keep = {"old_path", "new_path", "new_file", "renamed_file", "deleted_file"}
        if level != "none":
            keep |= {"additions", "deletions"}
        if level == "full_patch":
            keep.add("diff")
        result["diffs"] = [{k: v for k, v in d.items() if k in keep} for d in diffs]
        result["diffs_page_info"] = {
            "hasNextPage": start + size < len(record["_diffs"]),
            "endCursor": cursor(start + len(diffs)) if diffs else None,
        }
    elif facet == "commits":
        result["commits"] = [
            {"id": record["sha"], "short_id": record["sha"][:8], "title": record["title"]}
        ]
    elif facet == "notes":
        result["notes"] = []  # the data has no merge request notes
    elif facet is not None:
        raise ToolError(f"{facet}: the demo world has no merge request {facet}")
    return marshalled(result)


@server.tool(annotations=READ)
def list_merge_requests(
    url: str | None = None,
    project_id: str | None = None,
    group_id: str | None = None,
    author_username: str | None = None,
    assignee_username: str | None = None,
    reviewer_username: str | None = None,
    state: Literal["opened", "closed", "merged", "locked", "all"] | None = None,
    scope: Literal["created_by_me", "assigned_to_me", "review_requested"] | None = None,
    milestone: str | None = None,
    labels: str | None = None,
    search: str | None = None,
    after: str | None = None,
    first: First | None = None,
) -> CallToolResult:
    """A project's (or group's) merge requests, newest first, in a compact shape. scope needs a
    signed-in user, which the mock has not."""
    if scope:
        raise ToolError("scope needs a signed-in user; the demo world has none")
    if group_id and not project_id and not url:
        projects = [
            p for p in load_projects() if p.path.casefold().startswith(f"{group_id.casefold()}/")
        ]
    else:
        projects = [find_project(project_id, url)]
    terms = words(search or "")
    wanted = {label.strip() for label in (labels or "").split(",") if label.strip()}
    found = [
        (p, m)
        for p in projects
        for m in p.merge_requests
        if (not state or state == "all" or m["state"] == state)
        and (not terms or matches(f"{m['title']} {m.get('description') or ''}", terms))
        and (not author_username or m["author"]["username"] == author_username)
        and (
            not assignee_username
            or assignee_username in {a["username"] for a in m.get("assignees", [])}
        )
        and (not reviewer_username or reviewer_username in {r["username"] for r in m["reviewers"]})
        and (not wanted or wanted <= set(m.get("labels", [])))
        and not milestone
    ]
    found.sort(key=lambda pm: pm[1]["created_at"], reverse=True)
    start = cursor_offset(after)
    size = first or 20
    shown = found[start : start + size]
    return marshalled(
        {
            "items": [compact_merge_request(p, m) for p, m in shown],
            "pageInfo": {
                "hasNextPage": start + size < len(found),
                "endCursor": cursor(start + len(shown)) if shown else None,
            },
        }
    )


def resolve_ref(project: Project, ref: str | None) -> str:
    """The commit a ref names: HEAD, the default branch or the head commit; else 404."""
    wanted = (ref or "HEAD").strip().removeprefix("refs/heads/")
    if wanted in ("HEAD", project.tree.branch) or (
        len(wanted) >= 7 and project.tree.sha.startswith(wanted.lower())
    ):
        return project.tree.sha
    raise ToolError(f"404 Commit Not Found: {ref}")


@server.tool(annotations=READ)
def get_repository_file(
    url: str | None = None,
    project_id: str | None = None,
    file_path: str | None = None,
    ref: str | None = None,
    offset: Annotated[int, Field(ge=0)] | None = None,
    limit: Annotated[int, Field(ge=1, le=MAX_FILE_LINES)] | None = None,
) -> CallToolResult:
    """One text file at a ref (HEAD for the default branch), from line `offset` (0-based), at
    most `limit` lines (2000), with its metadata."""
    project = find_project(project_id, url)
    if url and not file_path and (m := re.search(r"/-/blob/([^/]+)/(.+)", urlparse(url).path)):
        ref, file_path = ref or m.group(1), unquote(m.group(2))
    if not file_path:
        raise ToolError("Provide file_path, ref and project_id, or a file url")
    commit = resolve_ref(project, ref)
    clean = file_path.strip().strip("/")
    if ".." in clean.split("/") or clean not in project.tree.files:
        raise ToolError(f"404 File Not Found: {file_path}")
    text = project.tree.files[clean]
    lines = text.split("\n")
    start = offset or 0
    size = limit or MAX_FILE_LINES
    shown = lines[start : start + size]
    return marshalled(
        {
            "file_path": clean,
            "file_name": clean.rsplit("/", 1)[-1],
            "ref": ref or "HEAD",
            "commit_id": commit,
            "encoding": "text",
            "content": "\n".join(shown),
            "total_lines": len(lines),
            "returned_lines": len(shown),
            "truncated": start + size < len(lines),
            "size_bytes": len(text.encode()),
        }
    )


@server.tool(annotations=READ)
def list_repository_tree(
    url: str | None = None,
    project_id: str | None = None,
    path: str | None = None,
    ref: str | None = None,
    recursive: bool | None = None,
    after: str | None = None,
) -> CallToolResult:
    """The files and directories at a path (metadata only)."""
    project = find_project(project_id, url)
    resolve_ref(project, ref)
    prefix = f"{(path or '').strip('/')}/" if (path or "").strip("/") else ""
    entries: dict[str, Record] = {}
    for file_path in project.tree.files:
        if not file_path.startswith(prefix):
            continue
        rest = file_path[len(prefix) :]
        parts = rest.split("/")
        if recursive:
            for depth in range(1, len(parts)):
                folder = prefix + "/".join(parts[:depth])
                entries.setdefault(
                    folder, {"name": parts[depth - 1], "type": "tree", "path": folder}
                )
            entries[file_path] = {"name": parts[-1], "type": "blob", "path": file_path}
        else:
            child = prefix + parts[0]
            kind = "tree" if len(parts) > 1 else "blob"
            entries.setdefault(child, {"name": parts[0], "type": kind, "path": child})
    if not entries:
        raise ToolError(f"404 Tree Not Found: {path}")
    return marshalled(sorted(entries.values(), key=lambda e: e["path"]))


@server.tool(annotations=READ)
def list_releases(
    url: str | None = None,
    project_id: str | None = None,
    page: Annotated[int, Field(ge=1)] | None = None,
    per_page: Annotated[int, Field(ge=1, le=100)] | None = None,
    state: Literal["released", "upcoming", "all"] | None = None,
) -> CallToolResult:
    """A project's releases, most recently released first: tag_name, name, released_at,
    upcoming and assets."""
    project = find_project(project_id, url)
    found = [
        r
        for r in project.releases
        if state in (None, "all") or (state == "upcoming") == r["upcoming"]
    ]
    found.sort(key=lambda r: r["released_at"], reverse=True)
    return marshalled(paged(found, page, per_page))


def main() -> None:
    settings = Settings()
    server.run("streamable-http", host=settings.world_mcp_host, port=settings.world_gitlab_mcp_port)
