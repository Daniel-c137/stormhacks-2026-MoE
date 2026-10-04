from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Sensitivity = Literal["quiet", "balanced", "eager"]
WhoCanAllow = Literal["everyone", "host"]
ConnectorName = Literal["github", "gitlab", "jira"]
ConnectorState = Literal["connected", "not_configured", "failing"]

MAX_CODE_REPOS = 10
"""Most repositories (GitHub) or projects (GitLab) one team connects, each."""


class CodeRepo(BaseModel):
    """One connected repository: a GitHub owner/name or a GitLab project path (group/project,
    subgroups included), read at `ref` (None: the default branch). Connection and index state
    belong to the server."""

    path: str
    ref: str | None = None
    connected: bool = False
    indexed_at: datetime | None = None
    files: int | None = None


class GitHubSettings(BaseModel):
    repos: list[CodeRepo] = Field(default=[], max_length=MAX_CODE_REPOS)  # owner/name each
    # The GitHub account whose token an admin connected (PUT /settings/github/account), by its
    # login: the team's reads then go to GitHub's hosted MCP server with that token. None: the
    # server's GITHUB_MCP_URL, with no credentials. The token never leaves the brain.
    account_login: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _single_repo(cls, data: Any) -> Any:
        """Settings saved before several repositories, {"repo", "ref", "connected",
        "indexed_at", "files"}, become a list of that one repository (none when repo is unset);
        what else they hold is ignored."""
        if not isinstance(data, dict) or "repos" in data or "repo" not in data:
            return data
        old = dict(data)
        repo = old.pop("repo")
        state = {k: old.pop(k) for k in ("ref", "connected", "indexed_at", "files") if k in old}
        if isinstance(repo, str) and repo.strip():
            return {**old, "repos": [{"path": repo.strip(), **state}]}
        return {**old, "repos": []}


class GitLabSettings(BaseModel):
    projects: list[CodeRepo] = Field(default=[], max_length=MAX_CODE_REPOS)  # group/project each


class JiraSettings(BaseModel):
    # The project the agent reads (answers, agenda suggestions, fact checks) and its site.
    site: str | None = None
    project: str | None = None
    # The account an admin connected for pushing (PUT /settings/jira/account): approved task
    # drafts become issues in `account_project` on `account_site`, as `account_email`. Its own
    # site and project, so connecting it changes nothing the agent reads. Its API token never
    # leaves the brain.
    connected: bool = False
    account_email: str | None = None
    account_site: str | None = None
    account_project: str | None = None


class Voice(BaseModel):
    id: str
    name: str
    desc: str
    sample: str
    default_label: str | None = None  # only on the agent's default voice, naming the agent


class TeamSettings(BaseModel):
    team_id: str
    # Connectors change only at PUT /settings/connectors; PUT /settings keeps them as saved.
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    gitlab: GitLabSettings = Field(default_factory=GitLabSettings)
    jira: JiraSettings = Field(default_factory=JiraSettings)
    voice: str | None = None
    wake_phrase: str | None = None  # None means the identity's default wake phrase
    sensitivity: Sensitivity = "balanced"
    interrupt_minutes: int = 5
    who_can_allow: WhoCanAllow = "everyone"
    timezone: str = "UTC"  # IANA name, e.g. "America/Vancouver"; dates people see use it


class CodeRepoChoice(BaseModel):
    """A repository or project an admin connects: its path and, optionally, a branch or tag."""

    path: str
    ref: str | None = None


class JiraChoice(BaseModel):
    site: str | None = None
    project: str | None = None


class JiraAccountConnect(BaseModel):
    """PUT /settings/jira/account (admins only): the Jira Cloud site (name.atlassian.net), the
    Atlassian account's email and API token, and the project issues are created in. The brain
    checks them against Jira before saving; the token is stored encrypted and never returned."""

    site: str
    email: str
    api_token: str
    project: str


class GitHubAccountConnect(BaseModel):
    """PUT /settings/github/account (admins only): a fine-grained personal access token with
    read access to the team's repositories. The brain checks it with GitHub before saving; it is
    stored encrypted and never returned."""

    token: str


class ConnectorsUpdate(BaseModel):
    """PUT /settings/connectors (admins only): the whole choice, replacing the saved one.
    Repositories already connected keep their connection and index state. The Jira account
    connected for pushing is separate and stays as it is."""

    github: list[CodeRepoChoice] = Field(default=[], max_length=MAX_CODE_REPOS)
    gitlab: list[CodeRepoChoice] = Field(default=[], max_length=MAX_CODE_REPOS)
    jira: JiraChoice = Field(default_factory=JiraChoice)


class ConnectorStatus(BaseModel):
    """Whether an integration can be used right now. Failing and unconfigured are never hidden."""

    name: ConnectorName
    state: ConnectorState
    detail: str | None = None
