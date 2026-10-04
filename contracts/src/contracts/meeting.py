from datetime import datetime
from typing import Literal

from pydantic import BaseModel

MeetingStatus = Literal["scheduled", "live", "processing", "needs_review", "pushed"]
ParticipantRole = Literal["host", "member"]


class Person(BaseModel):
    """A team member account."""

    id: str
    name: str
    short: str
    initials: str
    title: str | None = None
    photo_url: str | None = None


class Team(BaseModel):
    id: str
    name: str
    member_ids: list[str]
    github_repo: str | None = None
    jira_project: str | None = None


class Meeting(BaseModel):
    id: str
    team_id: str
    title: str
    status: MeetingStatus
    code: str
    host_id: str
    participant_ids: list[str]
    started_at: datetime | None = None
    scheduled_for: datetime | None = None  # set while status is scheduled
    duration_min: int | None = None
    jira_keys: list[str] = []


class Participant(BaseModel):
    """Live participant view; id is the account id and the LiveKit identity."""

    id: str
    name: str
    role: ParticipantRole
    is_agent: bool
    mic_on: bool
    cam_on: bool
    is_speaking: bool
    hand_raised: bool
