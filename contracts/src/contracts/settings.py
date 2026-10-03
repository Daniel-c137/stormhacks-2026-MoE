from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Sensitivity = Literal["quiet", "balanced", "eager"]
WhoCanAllow = Literal["everyone", "host"]


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


class TeamSettings(BaseModel):
    team_id: str
    github: GitHubSettings
    jira: JiraSettings
    voice: str | None = None
    wake_phrase: str | None = None  # None means the identity's default wake phrase
    sensitivity: Sensitivity = "balanced"
    interrupt_minutes: int = 5
    who_can_allow: WhoCanAllow = "everyone"
