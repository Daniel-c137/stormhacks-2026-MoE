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
    email: str | None = None
    photo_url: str | None = None
    is_admin: bool = False  # only an admin changes the connectors and creates accounts


class Team(BaseModel):
    id: str
    name: str
    member_ids: list[str]
    github_repo: str | None = None
    jira_project: str | None = None


class Meeting(BaseModel):
    """A scheduled meeting has scheduled_start and no started_at until someone starts it."""

    id: str
    team_id: str
    title: str
    status: MeetingStatus
    code: str
    host_id: str
    participant_ids: list[str]
    invitee_ids: list[str] = []
    scheduled_start: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_min: int | None = None
    jira_keys: list[str] = []
    transcript_deleted_at: datetime | None = None  # set once retention removed the segments
    agent_joined_at: datetime | None = None  # when the agent first joined; None if it never did


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
