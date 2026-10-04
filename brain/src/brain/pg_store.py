"""The Store on our Postgres 16 + pgvector. Schema: db/migrations (*_core.sql first)."""

from collections.abc import AsyncIterator, Collection, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from psycopg import AsyncCursor, errors
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from contracts import (
    Agenda,
    AgendaItem,
    ChatMessage,
    Decision,
    DecisionRelation,
    FactCheck,
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

from .db import connection
from .store import (
    Conflict,
    FactCheckState,
    GitHubAccount,
    JiraAccount,
    Login,
    NotFound,
    ReportAudio,
    default_settings,
    new_join_code,
    photo_path,
    unique_segments,
)

Row = dict[str, Any]
Cursor = AsyncCursor[Row]

MEETING = (
    "id, team_id, title, status, code, host_id, participant_ids, invitee_ids, scheduled_start,"
    " started_at, ended_at, duration_min, jira_keys, transcript_deleted_at, agent_joined_at,"
    " translate"
)
PERSON = "p.id, p.name, p.short, p.initials, p.title, p.email, p.photo_url, p.is_admin"
TEAM = """
select t.id, t.name, t.github_repo, t.jira_project,
    array(select m.person_id from memberships m where m.team_id = t.id order by m.seq)
        as member_ids
from teams t
"""
TASK = "id, meeting_id, title, description, owner_id, due, t, quote, include, key, url, jira_status"
DECISION = (
    "id, meeting_id, text, made_by, t, quote, chain, status, relation_type, relation_decision_id"
)
AGENDA = "generated_at, updated_at, current_item_id, tracked_until, revision"
SEGMENT = (
    "seg_id, meeting_id, speaker_id, speaker_name, text, is_final, t_start, t_end,"
    " language, original_text"
)
FACT_CHECK = (
    "id, claim, speaker_name, verdict, confidence, severity, finding, snippet_ids, sources, t,"
    " created_at"
)
FACT_CHECK_STATE = "meeting_id, checked_until, checked_at"
LOGIN = "person_id, email, password_hash, created_at, updated_at"
JIRA_ACCOUNT = (
    "team_id, site, project, issue_type_id, email, sealed_token, connected_by, connected_at"
)
GITHUB_ACCOUNT = "team_id, login, sealed_token, connected_by, connected_at"
# Newest first by started_at, or scheduled_start before it starts; ties in creation order.
NEWEST_MEETING_FIRST = "coalesce(m.started_at, m.scheduled_start) desc nulls last, m.seq"
REPORT_ROWS = ("meeting_id", "summary", "tasks", "decisions")


def _utc(row: Row) -> Row:
    """timestamptz comes back in the session's time zone; the API speaks UTC."""
    return {k: v.astimezone(UTC) if isinstance(v, datetime) else v for k, v in row.items()}


def _meeting(row: Row) -> Meeting:
    return Meeting.model_validate(_utc(row))


def _decision(row: Row) -> Decision:
    relation_type = row.pop("relation_type")
    related = row.pop("relation_decision_id")
    relation = DecisionRelation(type=relation_type, decision_id=related) if relation_type else None
    return Decision(**row, relation=relation)


def _columns(columns: str, alias: str) -> str:
    return ", ".join(f"{alias}.{c.strip()}" for c in columns.split(","))


def _from_excluded(*columns: str) -> str:
    """The SET list of an upsert that overwrites every given column."""
    names = [c.strip() for group in columns for c in group.split(",")]
    return ", ".join(f"{c} = excluded.{c}" for c in names if c != "id")


def _needle(query: str | None) -> str:
    return (query or "").strip().lower()


def _contains(column: str, param: str) -> str:
    """Case-insensitive substring match with no LIKE wildcards to escape."""
    return f"strpos(lower(coalesce({column}, '')), %({param})s) > 0"


class PostgresStore:
    """Every call is one transaction. Give it the app's pool, or a DSN to connect per call
    (scripts, and tests whose requests each run on their own event loop)."""

    def __init__(self, db: AsyncConnectionPool | str):
        self.db = db

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator[Cursor]:
        try:
            async with connection(self.db) as conn, conn.cursor(row_factory=dict_row) as cur:
                yield cur
        except errors.ForeignKeyViolation as e:
            # The row a write hangs off (team or meeting) is missing.
            raise NotFound(e.diag.message_detail or str(e)) from None

    async def _one(self, cur: Cursor, sql: str, params: Any = None) -> Row | None:
        await cur.execute(sql, params)
        return await cur.fetchone()

    async def _all(self, cur: Cursor, sql: str, params: Any = None) -> list[Row]:
        await cur.execute(sql, params)
        return await cur.fetchall()

    # teams and people

    async def create_team(self, team: Team) -> Team:
        async with self._tx() as cur:
            await cur.execute(
                "insert into teams (id, name, github_repo, jira_project)"
                " values (%(id)s, %(name)s, %(github_repo)s, %(jira_project)s)"
                " on conflict (id) do update set name = excluded.name,"
                " github_repo = excluded.github_repo, jira_project = excluded.jira_project",
                team.model_dump(),
            )
            await cur.execute("delete from memberships where team_id = %s", [team.id])
            await self._add_members(cur, team.id, team.member_ids)
            return await self._team(cur, team.id)

    async def team(self, team_id: str) -> Team:
        async with self._tx() as cur:
            return await self._team(cur, team_id)

    async def delete_team(self, team_id: str) -> None:
        async with self._tx() as cur:
            # Memberships, settings and meetings cascade, and everything under the meetings.
            row = await self._one(cur, "delete from teams where id = %s returning id", [team_id])
        if row is None:
            raise NotFound(f"team {team_id}")

    async def team_for_user(self, user_id: str) -> Team:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "select team_id from memberships where person_id = %s order by seq limit 1",
                [user_id],
            )
            if row is None:
                raise NotFound(f"user {user_id} has no team")
            return await self._team(cur, row["team_id"])

    async def upsert_person(self, person: Person, team_id: str) -> Person:
        async with self._tx() as cur:
            await self._team(cur, team_id)
            await cur.execute(
                "insert into people (id, name, short, initials, title, email, photo_url, is_admin)"
                " values (%(id)s, %(name)s, %(short)s, %(initials)s, %(title)s, %(email)s,"
                " %(photo_url)s, %(is_admin)s)"
                " on conflict (id) do update set name = excluded.name, short = excluded.short,"
                " initials = excluded.initials, title = excluded.title, email = excluded.email,"
                " photo_url = excluded.photo_url, is_admin = excluded.is_admin",
                person.model_dump(),
            )
            await self._add_members(cur, team_id, [person.id])
            return person.model_copy()

    async def person(self, person_id: str) -> Person:
        async with self._tx() as cur:
            row = await self._one(
                cur, f"select {PERSON} from people p where p.id = %s", [person_id]
            )
        if row is None:
            raise NotFound(f"person {person_id}")
        return Person.model_validate(row)

    async def update_person(self, person: Person) -> Person:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "update people p set name = %(name)s, short = %(short)s,"
                " initials = %(initials)s, title = %(title)s, email = %(email)s,"
                " photo_url = %(photo_url)s, is_admin = %(is_admin)s"
                f" where id = %(id)s returning {PERSON}",
                person.model_dump(),
            )
        if row is None:
            raise NotFound(f"person {person.id}")
        return Person.model_validate(row)

    async def save_photo(self, person_id: str, content_type: str, data: bytes) -> Person:
        async with self._tx() as cur:
            row = await self._set_photo_url(cur, person_id, photo_path(person_id, data))
            await cur.execute(
                "insert into person_photos (person_id, content_type, data) values (%s, %s, %s)"
                " on conflict (person_id) do update set content_type = excluded.content_type,"
                " data = excluded.data",
                [person_id, content_type, bytes(data)],
            )
        return Person.model_validate(row)

    async def photo(self, person_id: str) -> tuple[str, bytes]:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "select p.id, ph.content_type, ph.data from people p"
                " left join person_photos ph on ph.person_id = p.id where p.id = %s",
                [person_id],
            )
        if row is None:
            raise NotFound(f"person {person_id}")
        if row["data"] is None:
            raise NotFound(f"photo for person {person_id}")
        return row["content_type"], bytes(row["data"])

    async def delete_photo(self, person_id: str) -> Person:
        async with self._tx() as cur:
            row = await self._set_photo_url(cur, person_id, None)
            await cur.execute("delete from person_photos where person_id = %s", [person_id])
        return Person.model_validate(row)

    async def members(self, team_id: str) -> list[Person]:
        return await self._members(team_id, "", order="m.seq", limit=None)

    async def search_members(self, team_id: str, query: str, limit: int = 20) -> list[Person]:
        return await self._members(
            team_id, _needle(query), order='lower(p.name) collate "C", m.seq', limit=limit
        )

    async def settings(self, team_id: str) -> TeamSettings:
        async with self._tx() as cur:
            team = await self._team(cur, team_id)
            row = await self._one(
                cur,
                "select team_id, github, gitlab, jira, voice, wake_phrase, sensitivity,"
                " interrupt_minutes, who_can_allow, timezone from team_settings where team_id = %s",
                [team_id],
            )
        if row is None:
            return default_settings(team)
        return TeamSettings.model_validate(row)

    async def save_settings(self, settings: TeamSettings) -> TeamSettings:
        values = settings.model_dump(mode="json")
        values |= {key: Jsonb(values[key]) for key in ("github", "gitlab", "jira")}
        async with self._tx() as cur:
            await cur.execute(
                "insert into team_settings (team_id, github, gitlab, jira, voice, wake_phrase,"
                " sensitivity, interrupt_minutes, who_can_allow, timezone)"
                " values (%(team_id)s, %(github)s, %(gitlab)s, %(jira)s, %(voice)s,"
                " %(wake_phrase)s, %(sensitivity)s, %(interrupt_minutes)s, %(who_can_allow)s,"
                " %(timezone)s)"
                " on conflict (team_id) do update set github = excluded.github,"
                " gitlab = excluded.gitlab, jira = excluded.jira, voice = excluded.voice,"
                " wake_phrase = excluded.wake_phrase, sensitivity = excluded.sensitivity,"
                " interrupt_minutes = excluded.interrupt_minutes,"
                " who_can_allow = excluded.who_can_allow, timezone = excluded.timezone",
                values,
            )
        return settings.model_copy(deep=True)

    async def jira_account(self, team_id: str) -> JiraAccount | None:
        async with self._tx() as cur:
            row = await self._one(
                cur, f"select {JIRA_ACCOUNT} from jira_accounts where team_id = %s", [team_id]
            )
        return JiraAccount.model_validate(_utc(row)) if row else None

    async def save_jira_account(self, account: JiraAccount) -> JiraAccount:
        async with self._tx() as cur:
            await self._team(cur, account.team_id)
            await cur.execute(
                f"insert into jira_accounts ({JIRA_ACCOUNT})"
                " values (%(team_id)s, %(site)s, %(project)s, %(issue_type_id)s, %(email)s,"
                " %(sealed_token)s, %(connected_by)s, %(connected_at)s)"
                f" on conflict (team_id) do update set {_from_excluded(JIRA_ACCOUNT)}",
                account.model_dump(),
            )
        return account.model_copy()

    async def delete_jira_account(self, team_id: str) -> None:
        async with self._tx() as cur:
            await cur.execute("delete from jira_accounts where team_id = %s", [team_id])

    async def github_account(self, team_id: str) -> GitHubAccount | None:
        async with self._tx() as cur:
            row = await self._one(
                cur, f"select {GITHUB_ACCOUNT} from github_accounts where team_id = %s", [team_id]
            )
        return GitHubAccount.model_validate(_utc(row)) if row else None

    async def save_github_account(self, account: GitHubAccount) -> GitHubAccount:
        async with self._tx() as cur:
            await self._team(cur, account.team_id)
            await cur.execute(
                f"insert into github_accounts ({GITHUB_ACCOUNT})"
                " values (%(team_id)s, %(login)s, %(sealed_token)s, %(connected_by)s,"
                " %(connected_at)s)"
                f" on conflict (team_id) do update set {_from_excluded(GITHUB_ACCOUNT)}",
                account.model_dump(),
            )
        return account.model_copy()

    async def delete_github_account(self, team_id: str) -> None:
        async with self._tx() as cur:
            await cur.execute("delete from github_accounts where team_id = %s", [team_id])

    # logins

    async def set_login(self, person_id: str, email: str, password_hash: str) -> Login:
        try:
            async with self._tx() as cur:
                row = await self._one(
                    cur,
                    "insert into logins (person_id, email, password_hash)"
                    " values (%s, %s, %s)"
                    " on conflict (person_id) do update set email = excluded.email,"
                    " password_hash = excluded.password_hash, updated_at = now()"
                    f" returning {LOGIN}",
                    [person_id, email, password_hash],
                )
        except errors.UniqueViolation:
            raise Conflict("another person signs in with this email") from None
        return Login.model_validate(_utc(row or {}))

    async def add_login(self, person_id: str, email: str, password_hash: str) -> Login:
        async with self._tx() as cur:
            # no conflict target: neither the person's login nor the email's may exist yet
            row = await self._one(
                cur,
                "insert into logins (person_id, email, password_hash) values (%s, %s, %s)"
                f" on conflict do nothing returning {LOGIN}",
                [person_id, email, password_hash],
            )
        if row is None:
            raise Conflict("the person or the email has a login already")
        return Login.model_validate(_utc(row))

    async def login_by_email(self, email: str) -> Login:
        async with self._tx() as cur:
            row = await self._one(
                cur, f"select {LOGIN} from logins where lower(email) = lower(%s)", [email]
            )
        if row is None:
            raise NotFound("no login with this email")
        return Login.model_validate(_utc(row))

    async def login(self, person_id: str) -> Login:
        async with self._tx() as cur:
            row = await self._one(
                cur, f"select {LOGIN} from logins where person_id = %s", [person_id]
            )
        if row is None:
            raise NotFound(f"login for person {person_id}")
        return Login.model_validate(_utc(row))

    async def invited_people(self, email: str) -> list[Person]:
        async with self._tx() as cur:
            rows = await self._all(
                cur,
                f"select {PERSON} from people p"
                " where lower(btrim(p.email)) = lower(btrim(%s))"
                " and not exists (select 1 from logins l where l.person_id = p.id)"
                " and exists (select 1 from memberships m where m.person_id = p.id)"
                " order by p.id",
                [email],
            )
        return [Person.model_validate(r) for r in rows]

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
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "insert into meetings (id, team_id, title, status, code, host_id, invitee_ids,"
                " scheduled_start, started_at, duration_min, translate)"
                " values (%(id)s, %(team_id)s, %(title)s, %(status)s, %(code)s, %(host_id)s,"
                " %(invitee_ids)s, %(scheduled_start)s, %(started_at)s, %(duration_min)s,"
                " %(translate)s)"
                f" returning {MEETING}",
                {
                    "id": str(uuid4()),
                    "team_id": team_id,
                    "title": title,
                    "status": "scheduled" if scheduled_start else "live",
                    "code": new_join_code(),
                    "host_id": host_id,
                    "invitee_ids": list(dict.fromkeys(invitee_ids)),
                    "scheduled_start": scheduled_start,
                    "started_at": None if scheduled_start else datetime.now(UTC),
                    "duration_min": duration_min,
                    "translate": translate,
                },
            )
        assert row is not None
        return _meeting(row)

    async def meeting(self, meeting_id: str) -> Meeting:
        async with self._tx() as cur:
            return await self._meeting(cur, meeting_id)

    async def meeting_by_code(self, code: str) -> Meeting:
        async with self._tx() as cur:
            row = await self._one(cur, f"select {MEETING} from meetings where code = %s", [code])
        if row is None:
            raise NotFound(f"meeting code {code}")
        return _meeting(row)

    async def meetings(self, team_id: str) -> list[Meeting]:
        async with self._tx() as cur:
            rows = await self._all(
                cur,
                f"select {MEETING} from meetings m where team_id = %s"
                f" order by {NEWEST_MEETING_FIRST}",
                [team_id],
            )
        return [_meeting(r) for r in rows]

    async def set_status(self, meeting_id: str, status: MeetingStatus) -> Meeting:
        return await self._update_meeting(meeting_id, "status = %(status)s", status=status)

    async def transition_status(
        self,
        meeting_id: str,
        expected: Collection[MeetingStatus],
        to: MeetingStatus,
        *,
        at: datetime | None = None,
    ) -> Meeting:
        # One conditional UPDATE: the row lock makes overlapping calls queue, and each re-checks
        # the status the previous one left, so only one gets through.
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "update meetings set status = %(to)s::text,"
                " started_at = case when %(to)s::text = 'live' and status <> 'live'"
                "   then %(at)s else started_at end,"
                " ended_at = case when status = 'live' and %(to)s::text <> 'live'"
                "   then %(at)s else ended_at end"
                f" where id = %(id)s and status = any(%(expected)s::text[]) returning {MEETING}",
                {
                    "id": meeting_id,
                    "to": to,
                    "expected": list(expected),
                    "at": at or datetime.now(UTC),
                },
            )
            if row is None:
                current = await self._meeting(cur, meeting_id)
                raise Conflict(f"meeting {meeting_id} is {current.status}")
        return _meeting(row)

    async def start_meeting(self, meeting_id: str, at: datetime) -> Meeting:
        try:
            return await self.transition_status(meeting_id, {"scheduled"}, "live", at=at)
        except Conflict:
            return await self.meeting(meeting_id)

    async def add_participant(self, meeting_id: str, person_id: str) -> Meeting:
        # Computed from the row as it is when the update runs, so overlapping joins all land.
        return await self._update_meeting(
            meeting_id,
            "participant_ids = case when %(person)s::text = any(participant_ids)"
            " then participant_ids else array_append(participant_ids, %(person)s::text) end",
            person=person_id,
        )

    async def set_invitees(self, meeting_id: str, invitee_ids: Sequence[str]) -> Meeting:
        return await self._update_meeting(
            meeting_id, "invitee_ids = %(invitees)s", invitees=list(dict.fromkeys(invitee_ids))
        )

    async def update_meeting(self, meeting: Meeting) -> Meeting:
        return await self._update_meeting(
            meeting.id,
            "team_id = %(team_id)s, title = %(title)s, status = %(status)s, code = %(code)s,"
            " host_id = %(host_id)s, participant_ids = %(participant_ids)s,"
            " invitee_ids = %(invitee_ids)s, scheduled_start = %(scheduled_start)s,"
            " started_at = %(started_at)s, ended_at = %(ended_at)s,"
            " duration_min = %(duration_min)s, jira_keys = %(jira_keys)s,"
            " transcript_deleted_at = %(transcript_deleted_at)s,"
            " agent_joined_at = %(agent_joined_at)s, translate = %(translate)s",
            **meeting.model_dump(exclude={"id"}),
        )

    async def set_translate(self, meeting_id: str, on: bool) -> Meeting:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "update meetings set translate = %s where id = %s"
                " and status in ('scheduled', 'live') and cardinality(participant_ids) = 0"
                f" returning {MEETING}",
                [on, meeting_id],
            )
            if row is None:
                await self._meeting(cur, meeting_id)  # NotFound when there's no such meeting
                raise Conflict(f"meeting {meeting_id} can't change translation once someone joined")
        return _meeting(row)

    async def mark_agent_joined(self, meeting_id: str, at: datetime) -> Meeting:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "update meetings set agent_joined_at = coalesce(agent_joined_at, %s)"
                f" where id = %s and status = 'live' returning {MEETING}",
                [at, meeting_id],
            )
            if row is None:
                current = await self._meeting(cur, meeting_id)
                raise Conflict(f"meeting {meeting_id} is {current.status}")
        return _meeting(row)

    # transcript and public chat

    async def add_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        """An identical resend is ignored; a seg_id saved with different content is a Conflict.
        Checked after the insert in the same transaction, so it holds against a concurrent
        writer too, and a conflict rolls the whole batch back."""
        unique_segments(segments, {})
        async with self._tx() as cur:
            await self._meeting(cur, meeting_id)
            if not segments:
                return
            await cur.executemany(
                f"insert into transcript_segments ({SEGMENT})"
                " values (%(seg_id)s, %(meeting)s, %(speaker_id)s, %(speaker_name)s, %(text)s,"
                " %(is_final)s, %(t_start)s, %(t_end)s, %(language)s, %(original_text)s)"
                " on conflict (meeting_id, seg_id) do nothing",
                [s.model_dump() | {"meeting": meeting_id} for s in segments],
            )
            rows = await self._all(
                cur,
                f"select {SEGMENT} from transcript_segments"
                " where meeting_id = %s and seg_id = any(%s)",
                [meeting_id, [s.seg_id for s in segments]],
            )
            stored = {r["seg_id"]: TranscriptSegment.model_validate(r) for r in rows}
            unique_segments(segments, stored)

    async def transcript(self, meeting_id: str) -> list[TranscriptSegment]:
        async with self._tx() as cur:
            rows = await self._all(
                cur,
                f"select {SEGMENT} from transcript_segments where meeting_id = %s"
                " order by t_start, t_end, seg_id",
                [meeting_id],
            )
        return [TranscriptSegment.model_validate(r) for r in rows]

    async def delete_transcript(self, meeting_id: str, at: datetime) -> Meeting:
        async with self._tx() as cur:
            await cur.execute("delete from transcript_segments where meeting_id = %s", [meeting_id])
            row = await self._one(
                cur,
                f"update meetings set transcript_deleted_at = %s where id = %s returning {MEETING}",
                [at, meeting_id],
            )
        if row is None:
            raise NotFound(f"meeting {meeting_id}")
        return _meeting(row)

    async def meetings_with_transcript_before(self, cutoff: datetime) -> list[Meeting]:
        async with self._tx() as cur:
            rows = await self._all(
                cur,
                f"select {MEETING} from meetings m"
                " where ended_at < %s and transcript_deleted_at is null"
                " and exists (select 1 from transcript_segments s where s.meeting_id = m.id)"
                " order by ended_at, seq",
                [cutoff],
            )
        return [_meeting(r) for r in rows]

    async def add_public_chat(self, message: ChatMessage) -> None:
        if message.visibility != "public" or message.recipient_id is not None:
            raise ValueError("Private chat is never stored")
        async with self._tx() as cur:
            await cur.execute(
                "insert into public_chat (meeting_id, id, sender_id, sender_name, is_agent,"
                " text, ts, snippet_id)"
                " values (%(meeting_id)s, %(id)s, %(sender_id)s, %(sender_name)s, %(is_agent)s,"
                " %(text)s, %(ts)s, %(snippet_id)s)"
                " on conflict (meeting_id, id) do nothing",
                message.model_dump(),
            )

    async def public_chat(self, meeting_id: str) -> list[ChatMessage]:
        async with self._tx() as cur:
            rows = await self._all(
                cur,
                "select id, meeting_id, sender_id, sender_name, is_agent, text, ts, snippet_id"
                " from public_chat where meeting_id = %s order by ts, seq",
                [meeting_id],
            )
        return [ChatMessage.model_validate(_utc(r)) for r in rows]

    # agenda

    async def agenda(self, meeting_id: str) -> Agenda | None:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                f"select meeting_id, {AGENDA} from agendas where meeting_id = %s",
                [meeting_id],
            )
            if row is None:
                return None
            items = await self._all(
                cur,
                "select id, title, why, owner_id, sources, status, minutes, added_by,"
                " discussed_s, nudged_t, covered_by, covered_t from agenda_items"
                " where meeting_id = %s order by ord",
                [meeting_id],
            )
        return Agenda.model_validate(_utc(row) | {"items": items})

    async def save_agenda(self, agenda: Agenda) -> Agenda:
        async with self._tx() as cur:
            # One upsert: overlapping saves queue on the row lock and each bumps the revision.
            row = await self._one(
                cur,
                "insert into agendas (meeting_id, generated_at, updated_at, current_item_id,"
                " tracked_until, revision) values (%(meeting_id)s, %(generated_at)s,"
                " %(updated_at)s, %(current_item_id)s, %(tracked_until)s, 1)"
                " on conflict (meeting_id) do update set generated_at = excluded.generated_at,"
                " updated_at = excluded.updated_at, current_item_id = excluded.current_item_id,"
                " tracked_until = excluded.tracked_until, revision = agendas.revision + 1"
                " returning revision",
                agenda.model_dump(exclude={"items"}),
            )
            await self._replace_agenda_items(cur, agenda)
        return agenda.model_copy(deep=True, update={"revision": row["revision"]})

    async def save_agenda_if(self, agenda: Agenda) -> Agenda:
        if agenda.revision == 0:  # none saved yet: create it, unless someone just did
            sql = (
                "insert into agendas (meeting_id, generated_at, updated_at, current_item_id,"
                " tracked_until, revision) values (%(meeting_id)s, %(generated_at)s,"
                " %(updated_at)s, %(current_item_id)s, %(tracked_until)s, 1)"
                " on conflict (meeting_id) do nothing returning revision"
            )
        else:  # the row lock makes an overlapping save wait, then miss the old revision
            sql = (
                "update agendas set generated_at = %(generated_at)s,"
                " updated_at = %(updated_at)s, current_item_id = %(current_item_id)s,"
                " tracked_until = %(tracked_until)s, revision = revision + 1"
                " where meeting_id = %(meeting_id)s and revision = %(revision)s"
                " returning revision"
            )
        async with self._tx() as cur:
            row = await self._one(cur, sql, agenda.model_dump(exclude={"items"}))
            if row is None:
                await self._meeting(cur, agenda.meeting_id)  # NotFound when it is missing
                raise Conflict(f"agenda for meeting {agenda.meeting_id} changed meanwhile")
            await self._replace_agenda_items(cur, agenda)
        return agenda.model_copy(deep=True, update={"revision": row["revision"]})

    async def _replace_agenda_items(self, cur: Cursor, agenda: Agenda) -> None:
        await cur.execute("delete from agenda_items where meeting_id = %s", [agenda.meeting_id])
        if agenda.items:
            await cur.executemany(
                "insert into agenda_items (meeting_id, ord, id, title, why, owner_id,"
                " sources, status, minutes, added_by, discussed_s, nudged_t, covered_by,"
                " covered_t)"
                " values (%(meeting_id)s, %(ord)s, %(id)s, %(title)s, %(why)s, %(owner_id)s,"
                " %(sources)s, %(status)s, %(minutes)s, %(added_by)s, %(discussed_s)s,"
                " %(nudged_t)s, %(covered_by)s, %(covered_t)s)",
                [
                    self._agenda_item(agenda.meeting_id, i, item)
                    for i, item in enumerate(agenda.items)
                ],
            )

    # fact-checks

    async def add_fact_check(self, meeting_id: str, check: FactCheck) -> None:
        if check.recipient_id is not None:
            raise ValueError("Whom a fact-check was sent to is never stored")
        values = check.model_dump(mode="json") | {
            "meeting_id": meeting_id,
            "created_at": check.created_at,
            "snippet_ids": Jsonb(check.snippet_ids),
            "sources": Jsonb([s.model_dump(mode="json") for s in check.sources]),
        }
        async with self._tx() as cur:
            await cur.execute(
                f"insert into fact_checks (meeting_id, {FACT_CHECK})"
                " values (%(meeting_id)s, %(id)s, %(claim)s, %(speaker_name)s, %(verdict)s,"
                " %(confidence)s, %(severity)s, %(finding)s, %(snippet_ids)s, %(sources)s, %(t)s,"
                " %(created_at)s)"
                " on conflict (meeting_id, id) do nothing",
                values,
            )

    async def fact_checks(self, meeting_id: str) -> list[FactCheck]:
        async with self._tx() as cur:
            rows = await self._all(
                cur,
                f"select {FACT_CHECK} from fact_checks where meeting_id = %s order by seq",
                [meeting_id],
            )
        return [FactCheck.model_validate(_utc(r)) for r in rows]

    async def fact_check_state(self, meeting_id: str) -> FactCheckState | None:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                f"select {FACT_CHECK_STATE} from fact_check_state where meeting_id = %s",
                [meeting_id],
            )
        return FactCheckState.model_validate(row) if row else None

    async def save_fact_check_state_if(
        self, state: FactCheckState, *, checked_until: float | None
    ) -> FactCheckState:
        values = state.model_dump() | {"expected": checked_until}
        if checked_until is None:
            # A first save inserts; of overlapping first saves the later ones find the row.
            sql = (
                f"insert into fact_check_state ({FACT_CHECK_STATE})"
                " values (%(meeting_id)s, %(checked_until)s, %(checked_at)s)"
                " on conflict (meeting_id) do update set checked_until = excluded.checked_until,"
                " checked_at = excluded.checked_at"
                " where fact_check_state.checked_until is null"
                " returning meeting_id"
            )
        else:
            # The row lock makes an overlapping save wait, then find checked_until moved.
            sql = (
                "update fact_check_state set checked_until = %(checked_until)s,"
                " checked_at = %(checked_at)s"
                " where meeting_id = %(meeting_id)s and checked_until = %(expected)s"
                " returning meeting_id"
            )
        async with self._tx() as cur:
            if await self._one(cur, sql, values) is None:
                raise Conflict(f"fact-checks of meeting {state.meeting_id} moved on meanwhile")
        return state.model_copy(deep=True)

    # reports, tasks and decisions

    async def save_report(self, report: Report) -> None:
        async with self._tx() as cur:
            await self._write_report(cur, report)

    async def complete_report(self, report: Report, superseded: Sequence[Decision] = ()) -> Meeting:
        meeting_id = report.meeting_id
        async with self._tx() as cur:
            row = await self._one(
                cur, "select status from meetings where id = %s for update", [meeting_id]
            )
            if row is None:
                raise NotFound(f"meeting {meeting_id}")
            if row["status"] != "processing":
                raise Conflict(f"meeting {meeting_id} is {row['status']}")
            # Undo what this meeting's earlier decisions retired, before they are replaced.
            await cur.execute(
                "update decisions set status = 'active', relation_type = null,"
                " relation_decision_id = null"
                " where meeting_id <> %(m)s and relation_type = 'superseded_by'"
                " and relation_decision_id in (select id from decisions where meeting_id = %(m)s)",
                {"m": meeting_id},
            )
            await self._write_report(cur, report)
            for decision in superseded:
                found = await self._one(
                    cur,
                    "update decisions set status = %(status)s, relation_type = %(relation_type)s,"
                    " relation_decision_id = %(relation_decision_id)s"
                    " where id = %(id)s and meeting_id <> %(meeting)s returning id",
                    self._decision_values(decision) | {"meeting": meeting_id},
                )
                if found is None:
                    raise NotFound(f"decision {decision.id} of another meeting")
            row = await self._one(
                cur,
                f"update meetings set status = 'needs_review' where id = %s returning {MEETING}",
                [meeting_id],
            )
        assert row is not None  # the row is locked above, so it is still there
        return _meeting(row)

    async def _write_report(self, cur: Cursor, report: Report) -> None:
        """Replaces the meeting's report, tasks and decisions, in the caller's transaction."""
        meeting_id = report.meeting_id
        sections = report.model_dump(mode="json", exclude=set(REPORT_ROWS))
        await cur.execute(
            "insert into reports (meeting_id, summary, sections) values (%s, %s, %s)"
            " on conflict (meeting_id) do update set summary = excluded.summary,"
            " sections = excluded.sections",
            [meeting_id, report.summary, Jsonb(sections)],
        )
        await cur.execute("delete from task_drafts where meeting_id = %s", [meeting_id])
        await cur.execute("delete from decisions where meeting_id = %s", [meeting_id])
        if report.tasks:
            await cur.executemany(
                f"insert into task_drafts ({TASK}, ord)"
                " values (%(id)s, %(meeting_id)s, %(title)s, %(description)s, %(owner_id)s,"
                " %(due)s, %(t)s, %(quote)s, %(include)s, %(key)s, %(url)s, %(jira_status)s,"
                " %(ord)s)"
                f" on conflict (id) do update set {_from_excluded(TASK, 'ord')}",
                [t.model_dump() | {"ord": i} for i, t in enumerate(report.tasks)],
            )
        if report.decisions:
            await cur.executemany(
                f"insert into decisions ({DECISION}, ord)"
                " values (%(id)s, %(meeting_id)s, %(text)s, %(made_by)s, %(t)s, %(quote)s,"
                " %(chain)s, %(status)s, %(relation_type)s, %(relation_decision_id)s, %(ord)s)"
                f" on conflict (id) do update set {_from_excluded(DECISION, 'ord')}",
                [self._decision_values(d) | {"ord": i} for i, d in enumerate(report.decisions)],
            )

    async def report(self, meeting_id: str) -> Report:
        async with self._tx() as cur:
            row = await self._one(
                cur, "select summary, sections from reports where meeting_id = %s", [meeting_id]
            )
            if row is None:
                raise NotFound(f"report for meeting {meeting_id}")
            tasks = await self._all(
                cur,
                f"select {TASK} from task_drafts where meeting_id = %s order by ord",
                [meeting_id],
            )
            decisions = await self._all(
                cur,
                f"select {DECISION} from decisions where meeting_id = %s order by ord",
                [meeting_id],
            )
        return Report.model_validate(
            {
                **row["sections"],
                "meeting_id": meeting_id,
                "summary": row["summary"],
                "tasks": [TaskDraft.model_validate(t) for t in tasks],
                "decisions": [_decision(d) for d in decisions],
            }
        )

    async def save_report_progress(self, progress: ReportProgress) -> ReportProgress:
        async with self._tx() as cur:
            await cur.execute(
                "insert into report_progress"
                " (meeting_id, steps, current_step, done, error, updated_at)"
                " values (%s, %s, %s, %s, %s, %s)"
                " on conflict (meeting_id) do update set steps = excluded.steps,"
                " current_step = excluded.current_step, done = excluded.done,"
                " error = excluded.error, updated_at = excluded.updated_at",
                [
                    progress.meeting_id,
                    Jsonb(progress.steps),
                    progress.current,
                    progress.done,
                    progress.error,
                    progress.updated_at,
                ],
            )
        return progress.model_copy(deep=True)

    async def report_progress(self, meeting_id: str) -> ReportProgress | None:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "select meeting_id, steps, current_step as current, done, error, updated_at"
                " from report_progress where meeting_id = %s",
                [meeting_id],
            )
        return ReportProgress.model_validate(row) if row else None

    async def save_report_audio(
        self, meeting_id: str, key: str, content_type: str, data: bytes
    ) -> ReportAudio:
        async with self._tx() as cur:
            await cur.execute(
                "insert into report_audio (meeting_id, key, content_type, data)"
                " values (%s, %s, %s, %s) on conflict (meeting_id) do update"
                " set key = excluded.key, content_type = excluded.content_type,"
                " data = excluded.data",
                [meeting_id, key, content_type, bytes(data)],
            )
        return ReportAudio(key, content_type, bytes(data))

    async def report_audio(self, meeting_id: str) -> ReportAudio:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "select key, content_type, data from report_audio where meeting_id = %s",
                [meeting_id],
            )
        if row is None:
            raise NotFound(f"report audio for meeting {meeting_id}")
        return ReportAudio(row["key"], row["content_type"], bytes(row["data"]))

    async def task(self, team_id: str, task_id: str) -> TaskDraft:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                f"select {_columns(TASK, 't')}"
                " from task_drafts t join meetings m on m.id = t.meeting_id"
                " where t.id = %s and m.team_id = %s",
                [task_id, team_id],
            )
        if row is None:
            raise NotFound(f"task {task_id}")
        return TaskDraft.model_validate(row)

    async def update_task(self, task: TaskDraft) -> TaskDraft:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "update task_drafts set meeting_id = %(meeting_id)s, title = %(title)s,"
                " description = %(description)s, owner_id = %(owner_id)s, due = %(due)s,"
                " t = %(t)s, quote = %(quote)s, include = %(include)s, key = %(key)s,"
                f" url = %(url)s, jira_status = %(jira_status)s where id = %(id)s returning {TASK}",
                task.model_dump(),
            )
        if row is None:
            raise NotFound(f"task {task.id}")
        return TaskDraft.model_validate(row)

    async def tasks(self, team_id: str, owner_id: str | None = None) -> list[TaskDraft]:
        async with self._tx() as cur:
            rows = await self._all(
                cur,
                f"select {_columns(TASK, 't')}"
                " from task_drafts t join meetings m on m.id = t.meeting_id"
                " where m.team_id = %(team)s"
                " and (%(owner)s::text is null or t.owner_id = %(owner)s)"
                f" order by {NEWEST_MEETING_FIRST}, t.ord",
                {"team": team_id, "owner": owner_id},
            )
        return [TaskDraft.model_validate(r) for r in rows]

    async def decisions(self, team_id: str, query: str | None = None) -> list[Decision]:
        async with self._tx() as cur:
            rows = await self._all(
                cur,
                f"select {_columns(DECISION, 'd')}"
                " from decisions d join meetings m on m.id = d.meeting_id"
                f" where m.team_id = %(team)s and {_contains('d.text', 'needle')}"
                f" order by {NEWEST_MEETING_FIRST}, d.t desc, d.ord",
                {"team": team_id, "needle": _needle(query)},
            )
        return [_decision(r) for r in rows]

    async def update_decision(self, decision: Decision) -> Decision:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                "update decisions set meeting_id = %(meeting_id)s, text = %(text)s,"
                " made_by = %(made_by)s, t = %(t)s, quote = %(quote)s, chain = %(chain)s,"
                " status = %(status)s,"
                " relation_type = %(relation_type)s,"
                " relation_decision_id = %(relation_decision_id)s"
                f" where id = %(id)s returning {DECISION}",
                self._decision_values(decision),
            )
        if row is None:
            raise NotFound(f"decision {decision.id}")
        return _decision(row)

    # internals

    async def _team(self, cur: Cursor, team_id: str) -> Team:
        row = await self._one(cur, f"{TEAM} where t.id = %s", [team_id])
        if row is None:
            raise NotFound(f"team {team_id}")
        return Team.model_validate(row)

    async def _add_members(self, cur: Cursor, team_id: str, person_ids: Sequence[str]) -> None:
        await cur.execute(
            "insert into memberships (team_id, person_id)"
            " select %s, p from unnest(%s::text[]) with ordinality as u(p, n) order by n"
            " on conflict do nothing",
            [team_id, list(person_ids)],
        )

    async def _set_photo_url(self, cur: Cursor, person_id: str, url: str | None) -> Row:
        row = await self._one(
            cur,
            f"update people p set photo_url = %s where id = %s returning {PERSON}",
            [url, person_id],
        )
        if row is None:
            raise NotFound(f"person {person_id}")
        return row

    async def _members(
        self, team_id: str, needle: str, *, order: str, limit: int | None
    ) -> list[Person]:
        async with self._tx() as cur:
            await self._team(cur, team_id)
            rows = await self._all(
                cur,
                f"select {PERSON} from memberships m join people p on p.id = m.person_id"
                f" where m.team_id = %(team)s and ({_contains('p.name', 'needle')}"
                f" or {_contains('p.email', 'needle')} or {_contains('p.title', 'needle')})"
                f" order by {order} limit %(limit)s",
                {"team": team_id, "needle": needle, "limit": limit},
            )
        return [Person.model_validate(r) for r in rows]

    async def _meeting(self, cur: Cursor, meeting_id: str) -> Meeting:
        row = await self._one(cur, f"select {MEETING} from meetings where id = %s", [meeting_id])
        if row is None:
            raise NotFound(f"meeting {meeting_id}")
        return _meeting(row)

    async def _update_meeting(self, meeting_id: str, assignments: str, **params: Any) -> Meeting:
        async with self._tx() as cur:
            row = await self._one(
                cur,
                f"update meetings set {assignments} where id = %(id)s returning {MEETING}",
                params | {"id": meeting_id},
            )
        if row is None:
            raise NotFound(f"meeting {meeting_id}")
        return _meeting(row)

    @staticmethod
    def _agenda_item(meeting_id: str, ord: int, item: AgendaItem) -> Row:
        values = item.model_dump(mode="json")
        return values | {"meeting_id": meeting_id, "ord": ord, "sources": Jsonb(values["sources"])}

    @staticmethod
    def _decision_values(decision: Decision) -> Row:
        values = decision.model_dump(exclude={"relation"})
        relation = decision.relation
        return values | {
            "chain": Jsonb(values["chain"]),
            "relation_type": relation.type if relation else None,
            "relation_decision_id": relation.decision_id if relation else None,
        }
