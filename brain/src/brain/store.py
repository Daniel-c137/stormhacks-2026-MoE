import secrets
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from contracts import (
    ChatMessage,
    Decision,
    Meeting,
    Person,
    Report,
    TaskDraft,
    Team,
    TeamSettings,
    TranscriptSegment,
)
from contracts.meeting import MeetingStatus


class NotFound(LookupError):
    """No such row, or not one the caller's team can see."""


class Store(Protocol):
    """Supabase Postgres. Every read is scoped to a team."""

    async def team_for_user(self, user_id: str) -> Team: ...
    async def members(self, team_id: str) -> list[Person]: ...
    async def settings(self, team_id: str) -> TeamSettings: ...
    async def save_settings(self, settings: TeamSettings) -> TeamSettings: ...

    async def create_meeting(self, team_id: str, title: str, host_id: str) -> Meeting: ...
    async def meeting(self, meeting_id: str) -> Meeting: ...
    async def meeting_by_code(self, code: str) -> Meeting: ...
    async def meetings(self, team_id: str) -> list[Meeting]: ...
    async def set_status(self, meeting_id: str, status: MeetingStatus) -> Meeting: ...
    async def add_participant(self, meeting_id: str, person_id: str) -> Meeting: ...

    async def add_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None: ...
    async def transcript(self, meeting_id: str) -> list[TranscriptSegment]: ...
    async def add_public_chat(self, message: ChatMessage) -> None: ...

    async def save_report(self, report: Report) -> None: ...
    async def report(self, meeting_id: str) -> Report: ...
    async def update_task(self, task: TaskDraft) -> TaskDraft: ...
    async def decisions(self, team_id: str, query: str | None = None) -> list[Decision]: ...
    async def tasks(self, team_id: str, owner_id: str | None = None) -> list[TaskDraft]: ...


def new_join_code() -> str:
    """Unguessable and URL-safe; the link is the only thing a teammate needs to join."""
    return secrets.token_urlsafe(9)


class InMemoryStore:
    """Teams, people and meetings in process memory. For tests and local dev before Supabase;
    everything is lost on restart."""

    def __init__(self, teams: list[Team], people: list[Person]):
        self._teams = {t.id: t for t in teams}
        self._people = {p.id: p for p in people}
        self._meetings: dict[str, Meeting] = {}

    async def team_for_user(self, user_id: str) -> Team:
        for team in self._teams.values():
            if user_id in team.member_ids:
                return team
        raise NotFound(f"user {user_id} has no team")

    async def members(self, team_id: str) -> list[Person]:
        team = self._teams.get(team_id)
        if team is None:
            raise NotFound(f"team {team_id}")
        return [self._people[i] for i in team.member_ids if i in self._people]

    async def create_meeting(self, team_id: str, title: str, host_id: str) -> Meeting:
        meeting = Meeting(
            id=str(uuid4()),
            team_id=team_id,
            title=title,
            status="live",
            code=new_join_code(),
            host_id=host_id,
            participant_ids=[],
            started_at=datetime.now(UTC),
        )
        self._meetings[meeting.id] = meeting
        return meeting

    async def meeting(self, meeting_id: str) -> Meeting:
        meeting = self._meetings.get(meeting_id)
        if meeting is None:
            raise NotFound(f"meeting {meeting_id}")
        return meeting

    async def meeting_by_code(self, code: str) -> Meeting:
        for meeting in self._meetings.values():
            if meeting.code == code:
                return meeting
        raise NotFound(f"meeting code {code}")

    async def meetings(self, team_id: str) -> list[Meeting]:
        found = [m for m in self._meetings.values() if m.team_id == team_id]
        return sorted(
            found, key=lambda m: m.started_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

    async def set_status(self, meeting_id: str, status: MeetingStatus) -> Meeting:
        meeting = (await self.meeting(meeting_id)).model_copy(update={"status": status})
        self._meetings[meeting_id] = meeting
        return meeting

    async def add_participant(self, meeting_id: str, person_id: str) -> Meeting:
        meeting = await self.meeting(meeting_id)
        if person_id not in meeting.participant_ids:
            meeting = meeting.model_copy(
                update={"participant_ids": [*meeting.participant_ids, person_id]}
            )
            self._meetings[meeting_id] = meeting
        return meeting
