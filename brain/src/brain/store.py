import hashlib
import secrets
from collections.abc import Collection, Iterable, Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel

from contracts import (
    Agenda,
    ChatMessage,
    Decision,
    GitHubSettings,
    JiraSettings,
    Meeting,
    Person,
    Report,
    ReportProgress,
    TaskDraft,
    Team,
    TeamSettings,
    TranscriptSegment,
)
from contracts.meeting import MeetingStatus


class NotFound(LookupError):
    """No such row, or not one the caller's team can see."""


class Conflict(Exception):
    """The row is not in the state the write expected, e.g. another call already ended it."""


class Store(Protocol):
    """Supabase Postgres. Reads keyed by team take the team id and return only that team's rows.
    Reads keyed by meeting (meeting, transcript, report, agenda, ...) return any meeting's rows;
    the API checks the meeting's team first (deps.team_meeting).

    Missing rows raise NotFound, including the meeting a write hangs off. Writes return what
    was saved. The contract is pinned down by brain/tests/test_store_contract.py, which every
    implementation must pass.
    """

    # teams and people

    async def create_team(self, team: Team) -> Team: ...
    async def team(self, team_id: str) -> Team: ...
    async def team_for_user(self, user_id: str) -> Team: ...
    async def upsert_person(self, person: Person, team_id: str) -> Person:
        """Save the person and add them to the team's members if they are not already."""
        ...

    async def person(self, person_id: str) -> Person: ...
    async def update_person(self, person: Person) -> Person: ...
    async def save_photo(self, person_id: str, content_type: str, data: bytes) -> Person:
        """Replaces the person's photo and sets photo_url to photo_path(person_id, data)."""
        ...

    async def photo(self, person_id: str) -> tuple[str, bytes]:
        """(content_type, data); NotFound when the person has none."""
        ...

    async def delete_photo(self, person_id: str) -> Person:
        """Clears photo_url; a person without a photo is returned unchanged."""
        ...

    async def members(self, team_id: str) -> list[Person]: ...
    async def search_members(self, team_id: str, query: str, limit: int = 20) -> list[Person]:
        """Case-insensitive match on name, email or title, by name. A blank query lists all."""
        ...

    async def settings(self, team_id: str) -> TeamSettings:
        """Saved settings, or defaults (the team's repo and project) when never saved."""
        ...

    async def save_settings(self, settings: TeamSettings) -> TeamSettings: ...

    # meetings

    async def create_meeting(
        self,
        team_id: str,
        title: str,
        host_id: str,
        *,
        scheduled_start: datetime | None = None,
        duration_min: int | None = None,
        invitee_ids: Sequence[str] = (),
    ) -> Meeting:
        """Scheduled when scheduled_start is given; otherwise live, started now."""
        ...

    async def meeting(self, meeting_id: str) -> Meeting: ...
    async def meeting_by_code(self, code: str) -> Meeting: ...
    async def meetings(self, team_id: str) -> list[Meeting]:
        """Newest first by started_at, or scheduled_start for a meeting not started yet."""
        ...

    async def set_status(self, meeting_id: str, status: MeetingStatus) -> Meeting:
        """Unconditional. Kept for existing callers; new code uses transition_status."""
        ...

    async def transition_status(
        self,
        meeting_id: str,
        expected: Collection[MeetingStatus],
        to: MeetingStatus,
        *,
        at: datetime | None = None,
    ) -> Meeting:
        """Atomic compare-and-set: Conflict unless the status is one of `expected`, so of
        overlapping calls only one gets through. Moving to live records started_at; moving off
        live records ended_at; both at `at`, or now."""
        ...

    async def start_meeting(self, meeting_id: str, at: datetime) -> Meeting:
        """scheduled -> live at `at`, through transition_status. Any other status is unchanged."""
        ...

    async def add_participant(self, meeting_id: str, person_id: str) -> Meeting:
        """Atomic set-union: overlapping joins never lose each other."""
        ...

    async def set_invitees(self, meeting_id: str, invitee_ids: Sequence[str]) -> Meeting:
        """Replaces the invitees, in order, without duplicates."""
        ...

    async def update_meeting(self, meeting: Meeting) -> Meeting:
        """Saves the whole meeting, e.g. jira_keys or ended_at."""
        ...

    # transcript and public chat

    async def add_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        """Final segments, idempotent by (meeting_id, seg_id): one already saved is ignored."""
        ...

    async def transcript(self, meeting_id: str) -> list[TranscriptSegment]:
        """Ordered by (t_start, t_end, seg_id); empty when there is none."""
        ...

    async def delete_transcript(self, meeting_id: str, at: datetime) -> Meeting:
        """Drops the segments and records transcript_deleted_at."""
        ...

    async def meetings_with_transcript_before(self, cutoff: datetime) -> list[Meeting]:
        """Across all teams, for retention: ended before cutoff, segments not yet deleted."""
        ...

    async def add_public_chat(self, message: ChatMessage) -> None:
        """Public messages only (ValueError otherwise); an id already saved is ignored."""
        ...

    async def public_chat(self, meeting_id: str) -> list[ChatMessage]:
        """In time order."""
        ...

    # agenda

    async def agenda(self, meeting_id: str) -> Agenda | None: ...
    async def save_agenda(self, agenda: Agenda) -> Agenda:
        """Replaces the meeting's agenda."""
        ...

    async def save_agenda_if(self, agenda: Agenda, *, tracked_until: float | None) -> Agenda:
        """Replaces the agenda only if the saved one is still tracked up to `tracked_until`:
        an atomic compare-and-set, so of overlapping timekeeping ticks or edits only one gets
        through and the others get Conflict. NotFound when no agenda is saved."""
        ...

    # reports, tasks and decisions

    async def save_report(self, report: Report) -> None:
        """Replaces the meeting's report, tasks and decisions."""
        ...

    async def report(self, meeting_id: str) -> Report:
        """With the tasks and decisions as they are now, edits included."""
        ...

    async def save_report_progress(self, progress: ReportProgress) -> ReportProgress: ...
    async def report_progress(self, meeting_id: str) -> ReportProgress | None: ...
    async def task(self, team_id: str, task_id: str) -> TaskDraft: ...
    async def update_task(self, task: TaskDraft) -> TaskDraft:
        """Saves an existing task. No team check: read it with task(team_id, ...) first."""
        ...

    async def tasks(self, team_id: str, owner_id: str | None = None) -> list[TaskDraft]:
        """Newest meeting first, each meeting's tasks in report order."""
        ...

    async def decisions(self, team_id: str, query: str | None = None) -> list[Decision]:
        """Newest first; a query matches the decision text, ignoring case."""
        ...

    async def update_decision(self, decision: Decision) -> Decision:
        """Saves an existing decision. No team check: the API checks the meeting's team."""
        ...


def photo_path(person_id: str, data: bytes) -> str:
    """Where the API serves a person's photo. The version changes with the bytes, so a browser
    never shows a cached old photo."""
    version = hashlib.sha256(data).hexdigest()[:12]
    return f"/people/{person_id}/photo?v={version}"


def new_join_code() -> str:
    """Unguessable and URL-safe; the link is the only thing a teammate needs to join."""
    return secrets.token_urlsafe(9)


_NEVER = datetime.min.replace(tzinfo=UTC)


def _copy[M: BaseModel](model: M) -> M:
    """Callers get their own copy, as they would from a database."""
    return model.model_copy(deep=True)


def _when(meeting: Meeting) -> datetime:
    return meeting.started_at or meeting.scheduled_start or _NEVER


class InMemoryStore:
    """The whole store in process memory. For tests and local dev before Supabase; everything is
    lost on restart."""

    def __init__(self, teams: Iterable[Team] = (), people: Iterable[Person] = ()):
        self._teams = {t.id: _copy(t) for t in teams}
        self._people = {p.id: _copy(p) for p in people}
        self._photos: dict[str, tuple[str, bytes]] = {}
        self._settings: dict[str, TeamSettings] = {}
        self._meetings: dict[str, Meeting] = {}
        self._segments: dict[str, dict[str, TranscriptSegment]] = {}
        self._chat: dict[str, dict[str, ChatMessage]] = {}
        self._agendas: dict[str, Agenda] = {}
        self._reports: dict[str, Report] = {}
        self._progress: dict[str, ReportProgress] = {}
        self._tasks: dict[str, TaskDraft] = {}
        self._decisions: dict[str, Decision] = {}

    # teams and people

    async def create_team(self, team: Team) -> Team:
        self._teams[team.id] = _copy(team)
        return _copy(team)

    async def team(self, team_id: str) -> Team:
        return _copy(self._team(team_id))

    async def team_for_user(self, user_id: str) -> Team:
        for team in self._teams.values():
            if user_id in team.member_ids:
                return _copy(team)
        raise NotFound(f"user {user_id} has no team")

    async def upsert_person(self, person: Person, team_id: str) -> Person:
        team = self._team(team_id)
        self._people[person.id] = _copy(person)
        if person.id not in team.member_ids:
            team.member_ids = [*team.member_ids, person.id]
        return _copy(person)

    async def person(self, person_id: str) -> Person:
        return _copy(self._person(person_id))

    async def update_person(self, person: Person) -> Person:
        self._person(person.id)
        self._people[person.id] = _copy(person)
        return _copy(person)

    async def save_photo(self, person_id: str, content_type: str, data: bytes) -> Person:
        person = self._person(person_id)
        self._photos[person_id] = (content_type, bytes(data))
        return self._save_person(person, photo_url=photo_path(person_id, data))

    async def photo(self, person_id: str) -> tuple[str, bytes]:
        self._person(person_id)
        saved = self._photos.get(person_id)
        if saved is None:
            raise NotFound(f"photo for person {person_id}")
        return saved

    async def delete_photo(self, person_id: str) -> Person:
        person = self._person(person_id)
        self._photos.pop(person_id, None)
        return self._save_person(person, photo_url=None)

    async def members(self, team_id: str) -> list[Person]:
        team = self._team(team_id)
        return [_copy(self._people[i]) for i in team.member_ids if i in self._people]

    async def search_members(self, team_id: str, query: str, limit: int = 20) -> list[Person]:
        needle = query.strip().casefold()
        found = [
            p
            for p in await self.members(team_id)
            if any(needle in (field or "").casefold() for field in (p.name, p.email, p.title))
        ]
        return sorted(found, key=lambda p: p.name.casefold())[:limit]

    async def settings(self, team_id: str) -> TeamSettings:
        team = self._team(team_id)
        saved = self._settings.get(team_id)
        if saved is not None:
            return _copy(saved)
        return TeamSettings(
            team_id=team_id,
            github=GitHubSettings(repo=team.github_repo),
            jira=JiraSettings(project=team.jira_project),
        )

    async def save_settings(self, settings: TeamSettings) -> TeamSettings:
        self._team(settings.team_id)
        self._settings[settings.team_id] = _copy(settings)
        return _copy(settings)

    # meetings

    async def create_meeting(
        self,
        team_id: str,
        title: str,
        host_id: str,
        *,
        scheduled_start: datetime | None = None,
        duration_min: int | None = None,
        invitee_ids: Sequence[str] = (),
    ) -> Meeting:
        meeting = Meeting(
            id=str(uuid4()),
            team_id=team_id,
            title=title,
            status="scheduled" if scheduled_start else "live",
            code=new_join_code(),
            host_id=host_id,
            participant_ids=[],
            invitee_ids=list(dict.fromkeys(invitee_ids)),
            scheduled_start=scheduled_start,
            started_at=None if scheduled_start else datetime.now(UTC),
            duration_min=duration_min,
        )
        self._meetings[meeting.id] = meeting
        return _copy(meeting)

    async def meeting(self, meeting_id: str) -> Meeting:
        return _copy(self._meeting(meeting_id))

    async def meeting_by_code(self, code: str) -> Meeting:
        for meeting in self._meetings.values():
            if meeting.code == code:
                return _copy(meeting)
        raise NotFound(f"meeting code {code}")

    async def meetings(self, team_id: str) -> list[Meeting]:
        found = [_copy(m) for m in self._meetings.values() if m.team_id == team_id]
        return sorted(found, key=_when, reverse=True)

    async def set_status(self, meeting_id: str, status: MeetingStatus) -> Meeting:
        return self._save_meeting(self._meeting(meeting_id), status=status)

    async def transition_status(
        self,
        meeting_id: str,
        expected: Collection[MeetingStatus],
        to: MeetingStatus,
        *,
        at: datetime | None = None,
    ) -> Meeting:
        # No await between the check and the write, so this is atomic on the event loop.
        meeting = self._meeting(meeting_id)
        if meeting.status not in expected:
            raise Conflict(f"meeting {meeting_id} is {meeting.status}")
        when = at or datetime.now(UTC)
        changes: dict = {"status": to}
        if to == "live" and meeting.status != "live":
            changes["started_at"] = when
        elif meeting.status == "live" and to != "live":
            changes["ended_at"] = when
        return self._save_meeting(meeting, **changes)

    async def start_meeting(self, meeting_id: str, at: datetime) -> Meeting:
        try:
            return await self.transition_status(meeting_id, {"scheduled"}, "live", at=at)
        except Conflict:
            return await self.meeting(meeting_id)

    async def add_participant(self, meeting_id: str, person_id: str) -> Meeting:
        meeting = self._meeting(meeting_id)
        if person_id in meeting.participant_ids:
            return _copy(meeting)
        return self._save_meeting(meeting, participant_ids=[*meeting.participant_ids, person_id])

    async def set_invitees(self, meeting_id: str, invitee_ids: Sequence[str]) -> Meeting:
        meeting = self._meeting(meeting_id)
        return self._save_meeting(meeting, invitee_ids=list(dict.fromkeys(invitee_ids)))

    async def update_meeting(self, meeting: Meeting) -> Meeting:
        self._meeting(meeting.id)
        self._meetings[meeting.id] = _copy(meeting)
        return _copy(meeting)

    # transcript and public chat

    async def add_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        """Ignores a seg_id already saved: the worker may resend after a timeout."""
        self._meeting(meeting_id)
        saved = self._segments.setdefault(meeting_id, {})
        for segment in segments:
            saved.setdefault(segment.seg_id, _copy(segment))

    async def transcript(self, meeting_id: str) -> list[TranscriptSegment]:
        saved = self._segments.get(meeting_id, {}).values()
        return [_copy(s) for s in sorted(saved, key=lambda s: (s.t_start, s.t_end, s.seg_id))]

    async def delete_transcript(self, meeting_id: str, at: datetime) -> Meeting:
        meeting = self._meeting(meeting_id)
        self._segments.pop(meeting_id, None)
        return self._save_meeting(meeting, transcript_deleted_at=at)

    async def meetings_with_transcript_before(self, cutoff: datetime) -> list[Meeting]:
        found = [
            m
            for m in self._meetings.values()
            if m.ended_at is not None
            and m.ended_at < cutoff
            and m.transcript_deleted_at is None
            and self._segments.get(m.id)
        ]
        return [_copy(m) for m in sorted(found, key=lambda m: m.ended_at or _NEVER)]

    async def add_public_chat(self, message: ChatMessage) -> None:
        if message.visibility != "public" or message.recipient_id is not None:
            raise ValueError("Private chat is never stored")
        self._meeting(message.meeting_id)
        self._chat.setdefault(message.meeting_id, {}).setdefault(message.id, _copy(message))

    async def public_chat(self, meeting_id: str) -> list[ChatMessage]:
        saved = self._chat.get(meeting_id, {}).values()
        return [_copy(m) for m in sorted(saved, key=lambda m: m.ts)]

    # agenda

    async def agenda(self, meeting_id: str) -> Agenda | None:
        saved = self._agendas.get(meeting_id)
        return _copy(saved) if saved else None

    async def save_agenda(self, agenda: Agenda) -> Agenda:
        self._meeting(agenda.meeting_id)
        self._agendas[agenda.meeting_id] = _copy(agenda)
        return _copy(agenda)

    async def save_agenda_if(self, agenda: Agenda, *, tracked_until: float | None) -> Agenda:
        # No await between the check and the write, so this is atomic on the event loop.
        saved = self._agendas.get(agenda.meeting_id)
        if saved is None:
            raise NotFound(f"agenda for meeting {agenda.meeting_id}")
        if saved.tracked_until != tracked_until:
            raise Conflict(f"agenda for meeting {agenda.meeting_id} was tracked meanwhile")
        self._agendas[agenda.meeting_id] = _copy(agenda)
        return _copy(agenda)

    # reports, tasks and decisions

    async def save_report(self, report: Report) -> None:
        meeting_id = report.meeting_id
        self._meeting(meeting_id)
        for rows in (self._tasks, self._decisions):
            for row_id in [i for i, row in rows.items() if row.meeting_id == meeting_id]:
                del rows[row_id]
        self._tasks.update((t.id, _copy(t)) for t in report.tasks)
        self._decisions.update((d.id, _copy(d)) for d in report.decisions)
        self._reports[meeting_id] = _copy(report)

    async def report(self, meeting_id: str) -> Report:
        saved = self._reports.get(meeting_id)
        if saved is None:
            raise NotFound(f"report for meeting {meeting_id}")
        return saved.model_copy(
            update={
                "tasks": [_copy(self._tasks[t.id]) for t in saved.tasks],
                "decisions": [_copy(self._decisions[d.id]) for d in saved.decisions],
            },
            deep=True,
        )

    async def save_report_progress(self, progress: ReportProgress) -> ReportProgress:
        self._meeting(progress.meeting_id)
        self._progress[progress.meeting_id] = _copy(progress)
        return _copy(progress)

    async def report_progress(self, meeting_id: str) -> ReportProgress | None:
        saved = self._progress.get(meeting_id)
        return _copy(saved) if saved else None

    async def task(self, team_id: str, task_id: str) -> TaskDraft:
        task = self._tasks.get(task_id)
        if task is None or self._meeting(task.meeting_id).team_id != team_id:
            raise NotFound(f"task {task_id}")
        return _copy(task)

    async def update_task(self, task: TaskDraft) -> TaskDraft:
        if task.id not in self._tasks:
            raise NotFound(f"task {task.id}")
        self._tasks[task.id] = _copy(task)
        return _copy(task)

    async def tasks(self, team_id: str, owner_id: str | None = None) -> list[TaskDraft]:
        return [
            _copy(self._tasks[t.id])
            for report in self._team_reports(team_id)
            for t in report.tasks
            if owner_id is None or self._tasks[t.id].owner_id == owner_id
        ]

    async def decisions(self, team_id: str, query: str | None = None) -> list[Decision]:
        needle = (query or "").strip().casefold()
        found: list[Decision] = []
        for report in self._team_reports(team_id):
            current = [self._decisions[d.id] for d in report.decisions]
            found += sorted(
                (d for d in current if needle in d.text.casefold()),
                key=lambda d: d.t,
                reverse=True,
            )
        return [_copy(d) for d in found]

    async def update_decision(self, decision: Decision) -> Decision:
        if decision.id not in self._decisions:
            raise NotFound(f"decision {decision.id}")
        self._decisions[decision.id] = _copy(decision)
        return _copy(decision)

    # internals

    def _team(self, team_id: str) -> Team:
        team = self._teams.get(team_id)
        if team is None:
            raise NotFound(f"team {team_id}")
        return team

    def _person(self, person_id: str) -> Person:
        person = self._people.get(person_id)
        if person is None:
            raise NotFound(f"person {person_id}")
        return person

    def _save_person(self, person: Person, **changes) -> Person:
        saved = person.model_copy(update=changes, deep=True)
        self._people[person.id] = saved
        return _copy(saved)

    def _meeting(self, meeting_id: str) -> Meeting:
        meeting = self._meetings.get(meeting_id)
        if meeting is None:
            raise NotFound(f"meeting {meeting_id}")
        return meeting

    def _save_meeting(self, meeting: Meeting, **changes) -> Meeting:
        saved = meeting.model_copy(update=changes, deep=True)
        self._meetings[meeting.id] = saved
        return _copy(saved)

    def _team_reports(self, team_id: str) -> list[Report]:
        """The team's saved reports, newest meeting first."""
        meetings = [m for m in self._meetings.values() if m.team_id == team_id]
        return [
            self._reports[m.id]
            for m in sorted(meetings, key=_when, reverse=True)
            if m.id in self._reports
        ]
