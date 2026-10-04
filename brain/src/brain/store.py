import hashlib
import secrets
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import NamedTuple, Protocol
from uuid import uuid4

from pydantic import BaseModel

from contracts import (
    Agenda,
    ChatMessage,
    CodeRepo,
    Decision,
    FactCheck,
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


class ReportAudio(NamedTuple):
    """The report's summary read aloud. `key` names what it was made from (text, voice, model)."""

    key: str
    content_type: str
    data: bytes


class JiraAccount(BaseModel):
    """The Atlassian account an admin connected for a team's pushes, and the project its issues
    are created in. The API token is kept only sealed (brain.sealing); it never leaves the
    brain."""

    team_id: str
    site: str  # name.atlassian.net
    project: str
    issue_type_id: str | None = None  # the project's type tasks are created as; None: "Task"
    email: str
    sealed_token: str
    connected_by: str  # the admin's person id
    connected_at: datetime


class Login(BaseModel):
    """A person's email and password sign-in. Only the argon2 hash is kept; it never leaves the
    brain."""

    person_id: str
    email: str
    password_hash: str
    created_at: datetime
    updated_at: datetime


class FactCheckState(BaseModel):
    """Where a live meeting's fact-checks stand. Times are seconds from the meeting start."""

    meeting_id: str
    checked_until: float | None = None  # transcript checked so far; None before any
    checked_at: float | None = None  # the latest model check, for the rate limit


class Store(Protocol):
    """Our Postgres (pg_store.PostgresStore). Reads keyed by team take the team id and return
    only that team's rows. Reads keyed by meeting (meeting, transcript, report, agenda, ...)
    return any meeting's rows; the API checks the meeting's team first (deps.team_meeting).

    Missing rows raise NotFound, including the meeting a write hangs off. Writes return what
    was saved. The contract is pinned down by brain/tests/test_store_contract.py, which every
    implementation must pass.
    """

    # teams and people

    async def create_team(self, team: Team) -> Team: ...
    async def team(self, team_id: str) -> Team: ...
    async def delete_team(self, team_id: str) -> None:
        """Removes the team with its settings, memberships and meetings, and everything under
        them (transcripts, chat, agendas, fact-checks, reports, tasks, decisions, audio). Its
        people stay: a person is not the team's. NotFound for a missing team."""
        ...

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

    async def jira_account(self, team_id: str) -> JiraAccount | None:
        """The team's connected Jira account, or None."""
        ...

    async def save_jira_account(self, account: JiraAccount) -> JiraAccount:
        """Inserts or replaces the team's account. NotFound when the team is missing."""
        ...

    async def delete_jira_account(self, team_id: str) -> None:
        """Removing none is not an error."""
        ...

    # logins

    async def set_login(self, person_id: str, email: str, password_hash: str) -> Login:
        """Inserts or replaces the person's login. Conflict when another person's login has the
        email (ignoring case); NotFound when the person is missing."""
        ...

    async def add_login(self, person_id: str, email: str, password_hash: str) -> Login:
        """Inserts the person's first login and never replaces one: Conflict when the person
        already has a login or another login has the email (ignoring case); NotFound when the
        person is missing. Sign-up and Google claim an invited person's login with it."""
        ...

    async def login_by_email(self, email: str) -> Login:
        """Ignores case; NotFound when no login has the email."""
        ...

    async def login(self, person_id: str) -> Login:
        """NotFound when the person has no login."""
        ...

    async def invited_people(self, email: str) -> list[Person]:
        """People an admin invited with this email (ignoring case): on a team, with that email,
        and no login yet. Several when several teams invited it; the API refuses those."""
        ...

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
        translate: bool = False,
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

    async def set_translate(self, meeting_id: str, on: bool) -> Meeting:
        """Switches live translation (#106). Atomic, and only while the meeting is scheduled or
        live and nobody has joined: the worker reads it once when it opens the meeting.
        Conflict otherwise; NotFound for an unknown meeting."""
        ...

    async def mark_agent_joined(self, meeting_id: str, at: datetime) -> Meeting:
        """Records that the agent joined the live meeting at `at`, once: a later call keeps the
        first time. Atomic; Conflict unless the meeting is live."""
        ...

    # transcript and public chat

    async def add_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        """Final segments, idempotent by (meeting_id, seg_id): an identical resend is ignored,
        a seg_id reused with different content raises Conflict and nothing is saved."""
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
        """Replaces the meeting's agenda, whatever its revision, and bumps the revision.
        Returns what was saved, with the new revision."""
        ...

    async def save_agenda_if(self, agenda: Agenda) -> Agenda:
        """Replaces the agenda only if the saved one is still at `agenda.revision` (0: none saved
        yet), and bumps the revision: an atomic compare-and-set, so of overlapping writers that
        read the same revision (timekeeping ticks, lobby edits) only one gets through and the
        others get Conflict. NotFound when the meeting is missing."""
        ...

    # fact-checks

    async def add_fact_check(self, meeting_id: str, check: FactCheck) -> None:
        """The copy kept for the write-up: whom it was sent to is never stored, so a check with a
        recipient_id is refused (ValueError). An id already saved is ignored."""
        ...

    async def fact_checks(self, meeting_id: str) -> list[FactCheck]:
        """In the order they were added."""
        ...

    async def fact_check_state(self, meeting_id: str) -> FactCheckState | None: ...
    async def save_fact_check_state_if(
        self, state: FactCheckState, *, checked_until: float | None
    ) -> FactCheckState:
        """Saves the state only if the saved one is still checked up to `checked_until` (None
        when none is saved): an atomic compare-and-set, so of overlapping ticks only one gets
        through and the others get Conflict."""
        ...

    # reports, tasks and decisions

    async def save_report(self, report: Report) -> None:
        """Replaces the meeting's report, tasks and decisions."""
        ...

    async def complete_report(self, report: Report, superseded: Sequence[Decision] = ()) -> Meeting:
        """The write-up's final save, all or nothing: undoes the links this meeting's earlier
        decisions made (past decisions they retired become active again), replaces the report,
        tasks and decisions as save_report does, saves `superseded` (other meetings' decisions
        retired by this report's), and moves processing -> needs_review. Conflict unless the
        meeting is processing; NotFound for a missing meeting or a superseded decision that is
        not another meeting's."""
        ...

    async def report(self, meeting_id: str) -> Report:
        """With the tasks and decisions as they are now, edits included."""
        ...

    async def save_report_progress(self, progress: ReportProgress) -> ReportProgress: ...
    async def report_progress(self, meeting_id: str) -> ReportProgress | None: ...
    async def save_report_audio(
        self, meeting_id: str, key: str, content_type: str, data: bytes
    ) -> ReportAudio:
        """Replaces the meeting's stored summary audio."""
        ...

    async def report_audio(self, meeting_id: str) -> ReportAudio:
        """NotFound until audio is saved for the meeting."""
        ...

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


def default_settings(team: Team) -> TeamSettings:
    """A team's settings before any are saved: its own repository and Jira project, if any."""
    repos = [CodeRepo(path=team.github_repo)] if team.github_repo else []
    return TeamSettings(
        team_id=team.id,
        github=GitHubSettings(repos=repos),
        jira=JiraSettings(project=team.jira_project),
    )


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
    """The whole store in process memory. For tests and local dev without Postgres; everything is
    lost on restart."""

    def __init__(self, teams: Iterable[Team] = (), people: Iterable[Person] = ()):
        self._teams = {t.id: _copy(t) for t in teams}
        self._people = {p.id: _copy(p) for p in people}
        self._photos: dict[str, tuple[str, bytes]] = {}
        self._settings: dict[str, TeamSettings] = {}
        self._jira_accounts: dict[str, JiraAccount] = {}
        self._logins: dict[str, Login] = {}
        self._meetings: dict[str, Meeting] = {}
        self._segments: dict[str, dict[str, TranscriptSegment]] = {}
        self._chat: dict[str, dict[str, ChatMessage]] = {}
        self._agendas: dict[str, Agenda] = {}
        self._fact_checks: dict[str, dict[str, FactCheck]] = {}
        self._fact_check_states: dict[str, FactCheckState] = {}
        self._reports: dict[str, Report] = {}
        self._progress: dict[str, ReportProgress] = {}
        self._report_audio: dict[str, ReportAudio] = {}
        self._tasks: dict[str, TaskDraft] = {}
        self._decisions: dict[str, Decision] = {}

    # teams and people

    async def create_team(self, team: Team) -> Team:
        self._teams[team.id] = _copy(team)
        return _copy(team)

    async def team(self, team_id: str) -> Team:
        return _copy(self._team(team_id))

    async def delete_team(self, team_id: str) -> None:
        self._team(team_id)
        gone = {i for i, m in self._meetings.items() if m.team_id == team_id}
        for rows in (
            self._meetings,
            self._segments,
            self._chat,
            self._agendas,
            self._fact_checks,
            self._fact_check_states,
            self._reports,
            self._progress,
            self._report_audio,
        ):
            for meeting_id in gone & rows.keys():
                del rows[meeting_id]
        for rows in (self._tasks, self._decisions):
            for row_id in [i for i, row in rows.items() if row.meeting_id in gone]:
                del rows[row_id]
        self._settings.pop(team_id, None)
        self._jira_accounts.pop(team_id, None)
        del self._teams[team_id]

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
        return default_settings(team)

    async def save_settings(self, settings: TeamSettings) -> TeamSettings:
        self._team(settings.team_id)
        self._settings[settings.team_id] = _copy(settings)
        return _copy(settings)

    async def jira_account(self, team_id: str) -> JiraAccount | None:
        saved = self._jira_accounts.get(team_id)
        return _copy(saved) if saved else None

    async def save_jira_account(self, account: JiraAccount) -> JiraAccount:
        self._team(account.team_id)
        self._jira_accounts[account.team_id] = _copy(account)
        return _copy(account)

    async def delete_jira_account(self, team_id: str) -> None:
        self._jira_accounts.pop(team_id, None)

    # logins

    async def set_login(self, person_id: str, email: str, password_hash: str) -> Login:
        self._person(person_id)
        for other in self._logins.values():
            if other.person_id != person_id and other.email.lower() == email.lower():
                raise Conflict("another person signs in with this email")
        now = datetime.now(UTC)
        saved = self._logins.get(person_id)
        login = Login(
            person_id=person_id,
            email=email,
            password_hash=password_hash,
            created_at=saved.created_at if saved else now,
            updated_at=now,
        )
        self._logins[person_id] = login
        return _copy(login)

    async def add_login(self, person_id: str, email: str, password_hash: str) -> Login:
        self._person(person_id)
        if person_id in self._logins:
            raise Conflict("the person has a login already")
        return await self.set_login(person_id, email, password_hash)

    async def login_by_email(self, email: str) -> Login:
        for login in self._logins.values():
            if login.email.lower() == email.lower():
                return _copy(login)
        raise NotFound("no login with this email")

    async def login(self, person_id: str) -> Login:
        saved = self._logins.get(person_id)
        if saved is None:
            raise NotFound(f"login for person {person_id}")
        return _copy(saved)

    async def invited_people(self, email: str) -> list[Person]:
        wanted = email.strip().lower()
        on_a_team = {i for team in self._teams.values() for i in team.member_ids}
        return [
            _copy(p)
            for p in self._people.values()
            if p.id in on_a_team
            and p.id not in self._logins
            and (p.email or "").strip().lower() == wanted
        ]

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
        translate: bool = False,
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
            translate=translate,
        )
        self._meetings[meeting.id] = meeting
        return _copy(meeting)

    async def meeting(self, meeting_id: str) -> Meeting:
        return _copy(self._meeting(meeting_id))

    async def set_translate(self, meeting_id: str, on: bool) -> Meeting:
        # No await between the check and the write, so this is atomic on the event loop.
        meeting = self._meeting(meeting_id)
        if meeting.status not in ("scheduled", "live") or meeting.participant_ids:
            raise Conflict(f"meeting {meeting_id} can't change translation once someone joined")
        return self._save_meeting(meeting, translate=on)

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

    async def mark_agent_joined(self, meeting_id: str, at: datetime) -> Meeting:
        meeting = self._meeting(meeting_id)
        if meeting.status != "live":
            raise Conflict(f"meeting {meeting_id} is {meeting.status}")
        if meeting.agent_joined_at is not None:
            return _copy(meeting)
        return self._save_meeting(meeting, agent_joined_at=at)

    # transcript and public chat

    async def add_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        self._meeting(meeting_id)
        saved = self._segments.setdefault(meeting_id, {})
        unique_segments(segments, saved)
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
        return self._put_agenda(agenda)

    async def save_agenda_if(self, agenda: Agenda) -> Agenda:
        # No await between the check and the write, so this is atomic on the event loop.
        self._meeting(agenda.meeting_id)
        saved = self._agendas.get(agenda.meeting_id)
        if (saved.revision if saved else 0) != agenda.revision:
            raise Conflict(f"agenda for meeting {agenda.meeting_id} changed meanwhile")
        return self._put_agenda(agenda)

    def _put_agenda(self, agenda: Agenda) -> Agenda:
        saved = self._agendas.get(agenda.meeting_id)
        revision = (saved.revision if saved else 0) + 1
        self._agendas[agenda.meeting_id] = agenda.model_copy(
            deep=True, update={"revision": revision}
        )
        return _copy(self._agendas[agenda.meeting_id])

    # fact-checks

    async def add_fact_check(self, meeting_id: str, check: FactCheck) -> None:
        if check.recipient_id is not None:
            raise ValueError("Whom a fact-check was sent to is never stored")
        self._meeting(meeting_id)
        self._fact_checks.setdefault(meeting_id, {}).setdefault(check.id, _copy(check))

    async def fact_checks(self, meeting_id: str) -> list[FactCheck]:
        return [_copy(c) for c in self._fact_checks.get(meeting_id, {}).values()]

    async def fact_check_state(self, meeting_id: str) -> FactCheckState | None:
        saved = self._fact_check_states.get(meeting_id)
        return _copy(saved) if saved else None

    async def save_fact_check_state_if(
        self, state: FactCheckState, *, checked_until: float | None
    ) -> FactCheckState:
        # No await between the check and the write, so this is atomic on the event loop.
        self._meeting(state.meeting_id)
        saved = self._fact_check_states.get(state.meeting_id)
        if (saved.checked_until if saved else None) != checked_until:
            raise Conflict(f"fact-checks of meeting {state.meeting_id} moved on meanwhile")
        self._fact_check_states[state.meeting_id] = _copy(state)
        return _copy(state)

    # reports, tasks and decisions

    async def save_report(self, report: Report) -> None:
        self._meeting(report.meeting_id)
        self._write_report(report)

    async def complete_report(self, report: Report, superseded: Sequence[Decision] = ()) -> Meeting:
        # Every check comes before the first write, and nothing awaits in between.
        meeting = self._meeting(report.meeting_id)
        if meeting.status != "processing":
            raise Conflict(f"meeting {meeting.id} is {meeting.status}")
        for decision in superseded:
            saved = self._decisions.get(decision.id)
            if saved is None or saved.meeting_id == meeting.id:
                raise NotFound(f"decision {decision.id} of another meeting")
        earlier = {i for i, d in self._decisions.items() if d.meeting_id == meeting.id}
        for i, d in list(self._decisions.items()):
            if (
                d.meeting_id != meeting.id
                and d.relation is not None
                and d.relation.type == "superseded_by"
                and d.relation.decision_id in earlier
            ):
                self._decisions[i] = d.model_copy(update={"status": "active", "relation": None})
        self._write_report(report)
        self._decisions.update((d.id, _copy(d)) for d in superseded)
        return self._save_meeting(meeting, status="needs_review")

    def _write_report(self, report: Report) -> None:
        meeting_id = report.meeting_id
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

    async def save_report_audio(
        self, meeting_id: str, key: str, content_type: str, data: bytes
    ) -> ReportAudio:
        self._meeting(meeting_id)
        audio = ReportAudio(key, content_type, bytes(data))
        self._report_audio[meeting_id] = audio
        return audio

    async def report_audio(self, meeting_id: str) -> ReportAudio:
        saved = self._report_audio.get(meeting_id)
        if saved is None:
            raise NotFound(f"report audio for meeting {meeting_id}")
        return saved

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


def unique_segments(
    segments: list[TranscriptSegment], saved: Mapping[str, TranscriptSegment]
) -> None:
    """Raises Conflict if a seg_id in the batch is already saved, or repeated within the batch,
    with different content. An identical resend is fine: the worker retries after a timeout."""
    batch: dict[str, TranscriptSegment] = {}
    for segment in segments:
        known = saved.get(segment.seg_id) or batch.get(segment.seg_id)
        if known is not None and known != segment:
            raise Conflict(f"seg_id {segment.seg_id} was already saved with different content")
        batch[segment.seg_id] = segment
