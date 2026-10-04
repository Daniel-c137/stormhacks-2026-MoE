from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Sensitivity = Literal["quiet", "balanced", "eager"]
WhoCanAllow = Literal["everyone", "host"]
ConnectorName = Literal["github", "jira"]
ConnectorState = Literal["connected", "not_configured", "failing"]


class GitHubSettings(BaseModel):
    repo: str | None = None
    ref: str | None = None
    connected: bool = False
    indexed_at: datetime | None = None
    files: int | None = None


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
    github: GitHubSettings
    jira: JiraSettings
    voice: str | None = None
    wake_phrase: str | None = None  # None means the identity's default wake phrase
    sensitivity: Sensitivity = "balanced"
    interrupt_minutes: int = 5
    who_can_allow: WhoCanAllow = "everyone"
    timezone: str = "UTC"  # IANA name, e.g. "America/Vancouver"; dates people see use it


class ConnectorStatus(BaseModel):
    """Whether an integration can be used right now. Failing and unconfigured are never hidden."""

    name: ConnectorName
    state: ConnectorState
    detail: str | None = None
