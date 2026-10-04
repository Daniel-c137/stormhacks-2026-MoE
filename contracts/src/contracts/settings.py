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
    site: str | None = None
    project: str | None = None
    connected: bool = False


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


class ConnectorsUpdate(BaseModel):
    """PUT /settings/connectors (admins only): the whole choice, replacing the saved one.
    Repositories already connected keep their connection and index state."""

    github: list[CodeRepoChoice] = Field(default=[], max_length=MAX_CODE_REPOS)
    gitlab: list[CodeRepoChoice] = Field(default=[], max_length=MAX_CODE_REPOS)
    jira: JiraChoice = Field(default_factory=JiraChoice)


class ConnectorStatus(BaseModel):
    """Whether an integration can be used right now. Failing and unconfigured are never hidden."""

    name: ConnectorName
    state: ConnectorState
    detail: str | None = None
