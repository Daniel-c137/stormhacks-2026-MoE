"""Read the team's GitHub repository's issues, pull requests and code through the GitHub MCP
server.

Tool names and arguments follow GitHub's official MCP server. A team whose admin connected a
token reads GitHub's hosted server with it (GITHUB_HOSTED_MCP_URL); any other team reads
GITHUB_MCP_URL with no credentials. Every call goes through the read allowlist.
"""

import re
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote, unquote, urlparse

import anyio
from mcp.types import BlobResourceContents, EmbeddedResource, TextResourceContents
from pydantic import BaseModel

from brain.integrations import McpReader, McpTarget, McpToolError, ToolRefused, search_words
from brain.jira import root_cause

GitHubKind = Literal["issue", "pr"]
CodeHost = Literal["github", "gitlab"]
MAX_BODY = 400
MAX_CODE_QUERY = 200  # the server takes 256 characters, the repo: qualifier included
GITHUB_WEB = "https://github.com"

# Code search operators; even as plain words they could negate or widen the repo: scope.
OPERATORS = {"AND", "OR", "NOT"}
# GitHub allows this many AND, OR and NOT operators in one search, so this many words in an OR.
MAX_ANY_WORDS = 6
# How many items of an OR search are ranked before the closest are kept.
ANY_WORDS_POOL = 30
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
# The official server names a file it read repo://owner/repo/sha/<commit>/contents/<path>, or
# names the ref instead of a commit when it did not resolve one.
PINNED_FILE = re.compile(r"repo://([^/]+)/([^/]+)/sha/([0-9a-f]{40})/contents/(.+)")


def split_words(word: str) -> list[str]:
    """A hyphenated word as its words ("Approve-check" as speech-to-text writes it); a key or
    version with a digit (DS-104, v0.9-rc) stays whole."""
    if "-" not in word or any(c.isdigit() for c in word):
        return [word]
    return [part for part in word.split("-") if part]


def plain_forms(word: str) -> list[str]:
    """The word and the forms it may be written in without its ending: approved, approve, fixes,
    fix. Rough on purpose; a form that is no word matches nothing."""
    w = word.casefold()
    forms = [w]
    if w.endswith("ed"):
        forms += [w[:-1], w[:-2]]
    elif w.endswith("ing"):
        forms += [w[:-3] + "e", w[:-3]]
    elif w.endswith("es"):
        forms += [w[:-2], w[:-1]]
    elif w.endswith("s") and not w.endswith("ss"):
        forms.append(w[:-1])
    return [f for f in dict.fromkeys(forms) if len(f) >= 3 or f == w]


def any_words(terms: list[str]) -> list[str]:
    """The words of an OR search: every term first, then their plain forms, up to
    MAX_ANY_WORDS."""
    words = [t.casefold() for t in terms]
    words += [f for t in terms for f in plain_forms(t)[1:]]
    return list(dict.fromkeys(words))[:MAX_ANY_WORDS]


def has_word(text: str, word: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE))


def closest(items: list["GitHubItem"], terms: list[str]) -> list["GitHubItem"]:
    """Items with the most of the terms (in any of their plain forms) in the title or body
    first, keeping the server's order among equals."""

    def score(item: GitHubItem) -> int:
        text = f"{item.title}\n{item.body or ''}"
        return sum(any(has_word(text, f) for f in plain_forms(t)) for t in terms)

    scored = [(score(item), i, item) for i, item in enumerate(items)]
    return [item for n, _, item in sorted(scored, key=lambda x: (-x[0], x[1])) if n]


class UnsafePath(ValueError):
    """A file path that could leave the repository: absolute, a URL, or with '..'."""


def safe_path(path: str) -> str:
    """A file path relative to the repository root, or UnsafePath."""
    clean = path.strip().removeprefix("./")
    parts = clean.split("/")
    if (
        not clean
        or clean.startswith("/")
        or "\\" in clean
        or "\0" in clean
        or ":" in parts[0]  # a URL scheme or a drive letter
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise UnsafePath(f"{path!r} is not a file path inside the repository")
    return clean


class CodeHit(BaseModel):
    """A file of the repository that code search matched."""

    path: str


class CodeFile(BaseModel):
    """A file's text as its code host returned it, at `ref`: the commit SHA when the server
    resolved one (pinned), otherwise the branch or tag it was read at."""

    repo: str  # owner/name on GitHub, the project path on GitLab
    path: str
    ref: str
    pinned: bool
    text: str
    host: CodeHost = "github"
    web: str = GITHUB_WEB  # the host's web address, e.g. https://gitlab.example.com

    def lines(self) -> list[str]:
        """The file's lines as GitHub numbers them: split at newlines only (a form feed is not
        a line break), without carriage returns or the empty line after a final newline."""
        lines = [line.removesuffix("\r") for line in self.text.split("\n")]
        if lines and not lines[-1]:
            lines.pop()
        return lines

    def url(self, start_line: int, end_line: int) -> str:
        """The lines on the code host; a permalink when the ref is a commit."""
        ref, path = quote(self.ref, safe="/"), quote(self.path, safe="/")
        web = self.web.rstrip("/")
        if self.host == "gitlab":
            return f"{web}/{self.repo}/-/blob/{ref}/{path}#L{start_line}-{end_line}"
        return f"{web}/{self.repo}/blob/{ref}/{path}#L{start_line}-L{end_line}"


def branch_name(ref: str | None) -> str:
    """What a link names for a ref the server did not resolve: the branch or tag, or HEAD for
    the default branch."""
    if not ref:
        return "HEAD"
    return ref.removeprefix("refs/heads/").removeprefix("refs/tags/")


class CodeReadError(RuntimeError):
    """A read from a code host (GitHub or GitLab) failed."""


class GitHubError(CodeReadError):
    """A GitHub read failed."""


class GitHubItem(BaseModel):
    kind: GitHubKind
    number: int
    title: str
    state: str | None = None
    merged: bool | None = None
    merged_at: datetime | None = None
    updated_at: datetime | None = None
    body: str | None = None
    url: str | None = None
    # A pull request's read only, when the server gives them: e.g. "2 passed, 1 failed (lint)"
    # and "sam approved, ada commented"
    checks: str | None = None
    reviews: str | None = None


class GitHubRelease(BaseModel):
    tag: str
    name: str | None = None
    published_at: datetime | None = None
    body: str | None = None
    url: str | None = None


class GitHubReader:
    host: CodeHost = "github"

    def __init__(self, repo: str, target: McpTarget, ref: str | None = None):
        owner, _, name = repo.strip().partition("/")
        if not owner or not name or "/" in name:
            raise ValueError(f"the repository must be owner/name, not {repo!r}")
        self.owner, self.repo = owner, name
        self.ref = (ref or "").strip() or None  # None: the default branch
        self.reader = McpReader("github", target)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    async def search(self, text: str, kind: GitHubKind, limit: int = 6) -> list[GitHubItem]:
        """Issues or pull requests of the repository matching the text.

        The server lets a repo:, org: or user: qualifier in the query replace the owner/repo
        scope, so only plain words are sent, and anything returned from elsewhere is dropped."""
        words = search_words(text)
        if not words:
            return []
        tool = "search_pull_requests" if kind == "pr" else "search_issues"
        data = await self._call(tool, {"query": words, "owner": self.owner, "repo": self.repo})
        found = self.found(data, kind, limit)
        terms = [t for w in words.split() if w.upper() not in OPERATORS for t in split_words(w)]
        if found or len(terms) < 2:
            return found
        # No item has every word, which a search phrased as speech often asks ("approved check
        # fix" for "Enforce the Approve check"): any of them will do, closest first.
        query = " OR ".join(any_words(terms))
        data = await self._call(tool, {"query": query, "owner": self.owner, "repo": self.repo})
        return closest(self.found(data, kind, ANY_WORDS_POOL), terms)[:limit]

    async def latest(self, kind: GitHubKind, limit: int = 6) -> list[GitHubItem]:
        """The repository's most recently updated issues or pull requests, open or closed. The
        query is only the kind's qualifier, written here; the server scopes it to the
        repository."""
        tool, qualifier = (
            ("search_pull_requests", "is:pr") if kind == "pr" else ("search_issues", "is:issue")
        )
        arguments = {"query": qualifier, "owner": self.owner, "repo": self.repo}
        data = await self._call(
            tool, arguments | {"sort": "updated", "order": "desc", "perPage": limit}
        )
        return self.found(data, kind, limit)

    def found(self, data: Any, kind: GitHubKind, limit: int) -> list[GitHubItem]:
        """A search's items of this repository; anything from elsewhere is dropped."""
        if isinstance(data, dict):
            data = data.get("items", data.get("result"))
        if isinstance(data, dict):
            data = data.get("items")
        items = (self.item(raw, kind) for raw in data if isinstance(raw, dict)) if data else ()
        return [item for item in items if item is not None and self.ours(item.url)][:limit]

    def ours(self, url: str | None) -> bool:
        """Whether a result's link is in this repository: /owner/repo/... on any GitHub host."""
        path = urlparse(url or "").path.casefold()
        return path.startswith(f"/{self.owner}/{self.repo}/".casefold())

    async def read(self, number: int, kind: GitHubKind) -> GitHubItem:
        """One issue or pull request of the repository."""
        if kind == "pr":
            tool, arguments = "pull_request_read", {"pullNumber": number}
        else:
            tool, arguments = "issue_read", {"issue_number": number}
        data = await self._call(
            tool, {"method": "get", "owner": self.owner, "repo": self.repo, **arguments}
        )
        if isinstance(data, dict) and "number" not in data and isinstance(data.get("result"), dict):
            data = data["result"]
        item = self.item(data, kind) if isinstance(data, dict) else None
        if item is None:
            raise GitHubError(f"GitHub returned nothing for {self.full_name}#{number}")
        if item.kind == "pr":
            item = item.model_copy(update=await self._checks_and_reviews(item.number))
        return item

    async def _checks_and_reviews(self, number: int) -> dict[str, str | None]:
        """A pull request's commit statuses, check runs and reviews, read at the same time. Each
        is left out when the server cannot give it (the demo's mock has no reviews or check
        runs, and a token without the permission is refused)."""
        found: dict[str, Any] = {}

        async def read(method: str) -> None:
            arguments = {"method": method, "owner": self.owner, "repo": self.repo}
            try:
                found[method] = await self.reader.call(
                    "pull_request_read", arguments | {"pullNumber": number}
                )
            except Exception:
                found[method] = None

        async with anyio.create_task_group() as group:
            for method in ("get_status", "get_check_runs", "get_reviews"):
                group.start_soon(read, method)
        return {
            "checks": checks_summary(found["get_status"], found["get_check_runs"]),
            "reviews": reviews_summary(found["get_reviews"]),
        }

    async def releases(self, limit: int = 5) -> list[GitHubRelease]:
        """The repository's latest releases, newest first. What a release contains is what is
        released: a pull request merged after the latest release is not released yet."""
        data = await self._call(
            "list_releases", {"owner": self.owner, "repo": self.repo, "perPage": limit}
        )
        if isinstance(data, dict):
            data = data.get("result", data.get("items", data.get("releases")))
        raw = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        found = [release(r) for r in raw]
        return [r for r in found if r is not None][:limit]

    async def search_code(self, text: str, limit: int = 5) -> list[CodeHit]:
        """Files of the repository whose code matches the text.

        Code search takes no owner/repo arguments: the scope is the repo: qualifier added here
        to plain words, which cannot carry qualifiers or operators of their own. Results from
        any other repository are dropped."""
        words = " ".join(w for w in search_words(text).split() if w not in OPERATORS)
        if len(words) > MAX_CODE_QUERY:
            words = words[:MAX_CODE_QUERY].rsplit(" ", 1)[0]
        if not words:
            return []
        query = f"{words} repo:{self.full_name}"
        data = await self._call("search_code", {"query": query, "perPage": limit})
        hits: list[CodeHit] = []
        for raw in (data.get("items") if isinstance(data, dict) else None) or ():
            if not isinstance(raw, dict) or not self.same_repo(raw.get("repository")):
                continue
            try:
                hit = CodeHit(path=safe_path(str(raw.get("path") or "")))
            except UnsafePath:
                continue
            if hit not in hits:
                hits.append(hit)
        return hits[:limit]

    def same_repo(self, name: Any) -> bool:
        return isinstance(name, str) and name.casefold() == self.full_name.casefold()

    async def read_file(self, path: str, ref: str | None = None) -> CodeFile:
        """One text file of the repository at `ref` (default: the team's ref, else the default
        branch), pinned to the commit the server resolved the ref to when it names one."""
        path = safe_path(path)
        ref = (ref or "").strip() or self.ref
        arguments: dict[str, Any] = {"owner": self.owner, "repo": self.repo, "path": path}
        if ref:
            arguments["sha" if COMMIT_SHA.fullmatch(ref) else "ref"] = ref
        try:
            blocks = await self.reader.content("get_file_contents", arguments)
        except ToolRefused:
            raise
        except McpToolError as e:
            raise GitHubError(str(e)) from e
        except Exception as e:
            raise GitHubError(f"GitHub MCP call failed: {root_cause(e)}") from e

        resources = [b.resource for b in blocks if isinstance(b, EmbeddedResource)]
        if any(isinstance(r, BlobResourceContents) for r in resources):
            raise GitHubError(f"{self.full_name}/{path} is not a text file")
        texts = [r for r in resources if isinstance(r, TextResourceContents)]
        if not texts:
            raise GitHubError(f"GitHub returned no file for {self.full_name}/{path}")
        pinned = PINNED_FILE.fullmatch(str(texts[0].uri))
        if pinned and not (
            self.same_repo(f"{pinned.group(1)}/{pinned.group(2)}")
            and unquote(pinned.group(4)) == path
        ):
            raise GitHubError(f"GitHub returned another file for {self.full_name}/{path}")
        return CodeFile(
            repo=self.full_name,
            path=path,
            ref=pinned.group(3) if pinned else branch_name(ref),
            pinned=bool(pinned),
            text=texts[0].text,
        )

    def item(self, raw: dict[str, Any], kind: GitHubKind) -> GitHubItem | None:
        number = raw.get("number")
        if not isinstance(number, int):
            return None
        if kind == "issue" and raw.get("pull_request"):
            kind = "pr"  # issue search can return pull requests too
        body = raw.get("body")
        return GitHubItem(
            kind=kind,
            number=number,
            title=str(raw.get("title") or "").strip(),
            state=raw.get("state") if isinstance(raw.get("state"), str) else None,
            merged=raw.get("merged") if isinstance(raw.get("merged"), bool) else None,
            merged_at=timestamp(raw.get("merged_at")),
            updated_at=timestamp(raw.get("updated_at")),
            body=body.strip()[:MAX_BODY] if isinstance(body, str) and body.strip() else None,
            url=raw.get("html_url") if isinstance(raw.get("html_url"), str) else None,
        )

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        try:
            return await self.reader.call(tool, arguments)
        except ToolRefused:
            raise
        except McpToolError as e:
            raise GitHubError(str(e)) from e
        except Exception as e:
            raise GitHubError(f"GitHub MCP call failed: {root_cause(e)}") from e


def timestamp(value: Any) -> datetime | None:
    """An ISO 8601 time from GitHub, or None when it is missing or unreadable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def release(raw: dict[str, Any]) -> GitHubRelease | None:
    tag = raw.get("tag_name")
    if not isinstance(tag, str) or not tag.strip():
        return None
    body = raw.get("body")
    return GitHubRelease(
        tag=tag.strip(),
        name=raw.get("name") if isinstance(raw.get("name"), str) else None,
        published_at=timestamp(raw.get("published_at")),
        body=body.strip()[:MAX_BODY] if isinstance(body, str) and body.strip() else None,
        url=raw.get("html_url") if isinstance(raw.get("html_url"), str) else None,
    )


PASSED = {"success", "neutral"}
FAILED = {"failure", "error", "cancelled", "timed_out", "action_required", "startup_failure"}
NAMED_FAILURES = 3


def checks_summary(status: Any, runs: Any) -> str | None:
    """Commit statuses and check runs counted as passed, failed (named) and pending; skipped
    ones are left out. None when there are none."""
    outcomes: list[tuple[str, str]] = []  # (outcome, name)
    for raw in listed(status, "statuses"):
        state = str(raw.get("state") or "").lower()
        if state:
            outcome = {"success": "passed", "pending": "pending"}.get(state, "failed")
            outcomes.append((outcome, str(raw.get("context") or "")))
    for raw in listed(runs, "check_runs"):
        name = str(raw.get("name") or "")
        conclusion = str(raw.get("conclusion") or "").lower()
        if raw.get("status") != "completed" or not conclusion:
            outcomes.append(("pending", name))
        elif conclusion in PASSED:
            outcomes.append(("passed", name))
        elif conclusion in FAILED:
            outcomes.append(("failed", name))
    if not outcomes:
        return None
    parts = []
    for outcome in ("passed", "failed", "pending"):
        names = [name for found, name in outcomes if found == outcome]
        if not names:
            continue
        part = f"{len(names)} {outcome}"
        if outcome == "failed" and (shown := [n for n in names if n][:NAMED_FAILURES]):
            part += f" ({', '.join(shown)})"
        parts.append(part)
    return ", ".join(parts)


def listed(data: Any, key: str) -> list[dict[str, Any]]:
    """The objects listed under `key` of a result, or none."""
    found = data.get(key) if isinstance(data, dict) else None
    return [raw for raw in found if isinstance(raw, dict)] if isinstance(found, list) else []


REVIEW_STATES = {
    "APPROVED": "approved",
    "CHANGES_REQUESTED": "requested changes",
    "COMMENTED": "commented",
    "DISMISSED": "review dismissed",
}


def reviews_summary(reviews: Any) -> str | None:
    """Each reviewer's latest review, in the order they first reviewed; a reviewer's comment
    after an approval or a change request does not replace it. None when there are none."""
    if isinstance(reviews, dict):  # a list may come wrapped as {"result": [...]}
        reviews = next((reviews[k] for k in ("result", "reviews", "items") if k in reviews), None)
    latest: dict[str, str] = {}
    for raw in reviews if isinstance(reviews, list) else ():
        if not isinstance(raw, dict):
            continue
        user = raw.get("user")
        login = user.get("login") if isinstance(user, dict) else user
        state = REVIEW_STATES.get(str(raw.get("state") or "").upper())
        if not isinstance(login, str) or not login or state is None:
            continue
        if state == "commented" and latest.get(login, "commented") != "commented":
            continue
        latest[login] = state
    return ", ".join(f"{login} {state}" for login, state in latest.items()) or None
