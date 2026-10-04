"""A team's own GitHub connections: the fine-grained personal access token an admin connected
each repository with in Settings, checked with GitHub's REST API when it is connected, and the
hosted MCP server that repository's reads then go to with it.

The token is sent only as a bearer header, to GITHUB_API_URL (when connecting) and
GITHUB_HOSTED_MCP_URL (when reading); redirects are not followed, and it is never logged."""

import re
from typing import Literal
from urllib.parse import quote

import anyio
import httpx

from .auth import signing_secret
from .config import Settings
from .integrations import McpEndpoint
from .sealing import Unsealable, unseal
from .store import GitHubAccount

TIMEOUT = 15.0
MAX_TOKEN = 255
# Fine-grained tokens start with github_pat_; a classic token (ghp_) is not limited to chosen
# repositories, so it is not taken.
FINE_GRAINED = re.compile(r"github_pat_[A-Za-z0-9_]{20,240}")
NOT_FINE_GRAINED = (
    "Paste a fine-grained personal access token (it starts with github_pat_), with read access "
    "to the repository"
)
RATE_LIMITED = "GitHub's rate limit was reached: try again in a few minutes"
CONNECT_AGAIN = (
    "the token it was connected with can't be read on this server: an admin must connect it again"
)

Part = Literal["metadata", "issues", "pull_requests", "contents"]
# What the brain reads, by the permission a fine-grained token needs for it, and where to ask.
PARTS: dict[Part, tuple[str, str]] = {
    "metadata": ("", "Metadata"),
    "issues": ("/issues?per_page=1", "Issues"),
    "pull_requests": ("/pulls?per_page=1", "Pull requests"),
    "contents": ("/contents/", "Contents"),
}


def fine_grained(token: str) -> bool:
    return FINE_GRAINED.fullmatch(token) is not None


class GitHubRejected(RuntimeError):
    """GitHub answered with an error status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class GitHubUnreachable(RuntimeError):
    """GitHub's API did not answer."""


class GitHubApi:
    """The few REST calls that check a token: whose it is, and whether it reads a repository's
    metadata, issues, pull requests and code."""

    def __init__(
        self, base_url: str, token: str, transport: httpx.AsyncBaseTransport | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.transport = transport

    def __repr__(self) -> str:
        return f"GitHubApi({self.base_url!r}, token=<hidden>)"

    async def _get(self, client: httpx.AsyncClient, path: str) -> httpx.Response:
        try:
            response = await client.get(self.base_url + path)
        except httpx.HTTPError as e:
            raise GitHubUnreachable(f"Could not reach GitHub: {type(e).__name__}") from None
        if response.status_code >= 500:
            raise GitHubUnreachable(f"GitHub answered {response.status_code}")
        if rate_limited(response):  # not the token's fault: no answer about it yet
            raise GitHubUnreachable(RATE_LIMITED)
        return response

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self.transport,
            timeout=TIMEOUT,
            follow_redirects=False,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    async def login(self) -> str:
        """The token's owner. GitHubRejected (401) when GitHub does not accept the token."""
        async with self._client() as client:
            response = await self._get(client, "/user")
        login = response.json().get("login") if response.status_code == 200 else None
        if not isinstance(login, str) or not login:
            raise GitHubRejected(response.status_code, message(response))
        return login

    async def unreadable(self, repo: str) -> str | None:
        """Why the token cannot read what the brain reads of `repo` (owner/name), naming the
        permission it lacks; None when it can. An empty repository has no code to read."""
        path = "/repos/" + quote(repo, safe="/")
        found: dict[Part, int | GitHubUnreachable] = {}

        async with self._client() as client:

            async def ask(part: Part) -> None:
                # Kept, not raised: the task group would wrap it in an ExceptionGroup.
                try:
                    found[part] = (await self._get(client, path + PARTS[part][0])).status_code
                except GitHubUnreachable as e:
                    found[part] = e

            async with anyio.create_task_group() as group:
                for part in PARTS:
                    group.start_soon(ask, part)

        for result in found.values():
            if isinstance(result, GitHubUnreachable):
                raise result
        if found["metadata"] == 401:
            raise GitHubRejected(401, "GitHub did not accept the token")
        if found["metadata"] != 200:
            return f"This token can't see {repo}: give it access to that repository"
        for part, (_, permission) in PARTS.items():
            status = found[part]
            if status == 200 or (part == "contents" and status == 404):  # 404: no commits yet
                continue
            if part == "issues" and status == 410:  # Issues turned off: nothing to read
                continue
            return (
                f"This token can't read {repo}'s {permission.lower()}: give it read access to "
                f"{permission}"
            )
        return None


def rate_limited(response: httpx.Response) -> bool:
    """GitHub's primary (no requests left) or secondary rate limit, which answer 403 or 429."""
    if response.status_code == 429:
        return True
    if response.status_code != 403:
        return False
    return (
        response.headers.get("x-ratelimit-remaining") == "0"
        or "rate limit" in message(response).lower()
    )


def message(response: httpx.Response) -> str:
    try:
        text = response.json().get("message")
    except ValueError:
        text = None
    return str(text or f"GitHub answered {response.status_code}")[:200]


def github_endpoints(
    settings: Settings, accounts: list[GitHubAccount]
) -> dict[str, McpEndpoint | str]:
    """Where each repository connected with a token is read, by its owner/name in lower case:
    GitHub's hosted MCP server with that token, or why the token can't be used. A repository
    with a token never falls back to the mock; one without is read from GITHUB_MCP_URL."""
    found: dict[str, McpEndpoint | str] = {}
    for account in accounts:
        token = unsealed_token(settings, account)
        found[account.repo] = (
            CONNECT_AGAIN if token is None else McpEndpoint(settings.github_hosted_mcp_url, token)
        )
    return found


class Unusable(RuntimeError):
    """A repository that can't be read here. `failing`: its token can't be opened; otherwise
    nothing to read it with is set up. `detail` says why, the message also names it."""

    def __init__(self, path: str, detail: str, *, failing: bool):
        prefix = "GitHub is not reachable" if failing else "GitHub is not configured"
        super().__init__(f"{prefix}: {path}: {detail}")
        self.detail = detail
        self.failing = failing


def repo_target(
    settings: Settings, path: str, endpoints: dict[str, McpEndpoint | str]
) -> McpEndpoint | str:
    """Where one repository (owner/name) is read: GitHub's hosted server with the token it was
    connected with, or GITHUB_MCP_URL with no credentials for the demo world's repositories
    (MOCK_GITHUB_OWNERS) and any connected without a token. Unusable when its token can't be
    opened or neither is there."""
    own = None if settings.mocks_repository(path) else endpoints.get(path.lower())
    if isinstance(own, str):
        raise Unusable(path, own, failing=True)
    if own is not None:
        return own
    if settings.github_mcp_url:
        return settings.github_mcp_url
    raise Unusable(
        path, "connect it with a token in Settings, or set GITHUB_MCP_URL", failing=False
    )


def unsealed_token(settings: Settings, account: GitHubAccount) -> str | None:
    """The account's token, or None when this server can't open it (AUTH_SECRET unset or
    changed since it was connected)."""
    secret = signing_secret(settings)
    if secret is None:
        return None
    try:
        return unseal(account.sealed_token, secret, account.team_id)
    except Unsealable:
        return None
