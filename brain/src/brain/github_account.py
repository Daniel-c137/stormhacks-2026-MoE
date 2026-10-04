"""A team's own GitHub connection: the fine-grained personal access token an admin connected in
Settings, checked with GitHub's REST API when it is connected, and the hosted MCP server the
team's reads then go to with it.

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
    "to the team's repositories"
)
CONNECT_AGAIN = (
    "the connected token can't be read on this server: an admin must connect GitHub again"
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
        found: dict[Part, int] = {}

        async with self._client() as client:

            async def ask(part: Part) -> None:
                found[part] = (await self._get(client, path + PARTS[part][0])).status_code

            async with anyio.create_task_group() as group:
                for part in PARTS:
                    group.start_soon(ask, part)

        if found["metadata"] == 401:
            raise GitHubRejected(401, "GitHub did not accept the token")
        if found["metadata"] != 200:
            return f"This token can't see {repo}: give it access to that repository"
        for part, (_, permission) in PARTS.items():
            status = found[part]
            if status == 200 or (part == "contents" and status == 404):  # 404: no commits yet
                continue
            return (
                f"This token can't read {repo}'s {permission.lower()}: give it read access to "
                f"{permission}"
            )
        return None


def message(response: httpx.Response) -> str:
    try:
        text = response.json().get("message")
    except ValueError:
        text = None
    return str(text or f"GitHub answered {response.status_code}")[:200]


def github_endpoint(settings: Settings, account: GitHubAccount | None) -> McpEndpoint | str | None:
    """Where a team reads GitHub: GitHub's hosted MCP server with its connected token; None
    when it has none (GITHUB_MCP_URL is read as before); or why its token can't be used. A
    team with a token never falls back to the mock."""
    if account is None:
        return None
    token = unsealed_token(settings, account)
    if token is None:
        return CONNECT_AGAIN
    return McpEndpoint(settings.github_hosted_mcp_url, token)


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
