"""Transcript retention: segments and their memory chunks go after the retention period; the
report, decisions and tasks stay."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from brain.config import Settings
from brain.db import migrate
from brain.llm.mock import MockEmbedder
from brain.memory import InMemoryMemoryStore, MeetingMemory, PgMemoryStore
from brain.pg_store import PostgresStore
from brain.retention import (
    RetentionUnavailable,
    open_retention_stores,
    purge_transcripts,
    transcripts_due,
)
from brain.store import InMemoryStore
from contracts import Person, Report, TaskDraft, Team, TranscriptSegment

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
ALEX = Person(id="u-alex", name="Alex Chen", short="Alex", initials="AC")
TEAM = Team(id="t-1", name="Checkout", member_ids=[ALEX.id])
OTHER = Team(id="t-2", name="Elsewhere", member_ids=[ALEX.id])


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(teams=[TEAM, OTHER], people=[ALEX])


@pytest.fixture
def memory() -> MeetingMemory:
    return MeetingMemory(MockEmbedder(dim=16), InMemoryMemoryStore())


def segments(meeting_id: str, n: int = 2) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            seg_id=f"{meeting_id}-{i}",
            meeting_id=meeting_id,
            speaker_id=ALEX.id,
            speaker_name=ALEX.name,
            text=f"Alex says thing {i} about the refund",
            is_final=True,
            t_start=i * 5.0,
            t_end=i * 5.0 + 2.5,
        )
        for i in range(n)
    ]


async def meeting_ended(
    store: InMemoryStore,
    ended: datetime,
    *,
    status: str = "needs_review",
    team: Team = TEAM,
    title: str = "Standup",
):
    """A meeting with a transcript that ended at `ended` and then moved to `status`."""
    meeting = await store.create_meeting(team.id, title, ALEX.id)
    await store.start_meeting(meeting.id, ended - timedelta(minutes=30))
    await store.add_segments(meeting.id, segments(meeting.id))
    if status != "live":
        await store.transition_status(meeting.id, {"live"}, "processing", at=ended)
    if status not in ("live", "processing"):
        await store.set_status(meeting.id, status)
    return await store.meeting(meeting.id)


def days_ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


# what is due


async def test_a_meeting_ended_before_the_cutoff_loses_its_transcript(store, memory):
    old = await meeting_ended(store, days_ago(15))

    result = await purge_transcripts(store, memory, now=NOW, retention_days=14)

    assert [m.id for m in result.deleted] == [old.id]
    assert result.deleted[0].transcript_deleted_at == NOW
    assert await store.transcript(old.id) == []
    assert (await store.meeting(old.id)).transcript_deleted_at == NOW
    assert result.failed == {}
    assert result.memory_cleared


async def test_the_cutoff_is_exactly_retention_days_before_now(store, memory):
    just_inside = await meeting_ended(store, days_ago(14))
    just_outside = await meeting_ended(store, days_ago(14) - timedelta(seconds=1))

    result = await purge_transcripts(store, memory, now=NOW, retention_days=14)

    assert [m.id for m in result.deleted] == [just_outside.id]
    assert len(await store.transcript(just_inside.id)) == 2
    assert (await store.meeting(just_inside.id)).transcript_deleted_at is None


async def test_every_team_is_purged(store, memory):
    mine = await meeting_ended(store, days_ago(20))
    theirs = await meeting_ended(store, days_ago(20), team=OTHER)

    result = await purge_transcripts(store, memory, now=NOW, retention_days=14)

    assert {m.id for m in result.deleted} == {mine.id, theirs.id}


async def test_pushed_meetings_are_purged_too(store, memory):
    pushed = await meeting_ended(store, days_ago(20), status="pushed")

    result = await purge_transcripts(store, memory, now=NOW, retention_days=14)

    assert [m.id for m in result.deleted] == [pushed.id]


async def test_live_and_processing_meetings_are_never_purged(store, memory):
    processing = await meeting_ended(store, days_ago(30), status="processing")
    restarted = await meeting_ended(store, days_ago(30))
    # ended once long ago, then live again: ended_at is old but the meeting is running
    await store.set_status(restarted.id, "live")

    result = await purge_transcripts(store, memory, now=NOW, retention_days=14)

    assert result.deleted == []
    assert len(await store.transcript(processing.id)) == 2
    assert len(await store.transcript(restarted.id)) == 2


async def test_listing_what_is_due_changes_nothing(store):
    old = await meeting_ended(store, days_ago(15))
    await meeting_ended(store, days_ago(1))

    due = await transcripts_due(store, now=NOW, retention_days=14)

    assert [m.id for m in due] == [old.id]
    assert len(await store.transcript(old.id)) == 2
    assert (await store.meeting(old.id)).transcript_deleted_at is None


async def test_retention_must_be_at_least_a_day(store, memory):
    with pytest.raises(ValueError):
        await purge_transcripts(store, memory, now=NOW, retention_days=0)
    with pytest.raises(ValueError):
        await transcripts_due(store, now=NOW, retention_days=-3)


# safe to repeat


async def test_running_again_deletes_nothing_more_and_keeps_the_first_deletion_time(store, memory):
    old = await meeting_ended(store, days_ago(15))
    await purge_transcripts(store, memory, now=NOW, retention_days=14)

    again = await purge_transcripts(store, memory, now=NOW + timedelta(hours=1), retention_days=14)

    assert again.deleted == []
    assert again.failed == {}
    assert (await store.meeting(old.id)).transcript_deleted_at == NOW


# memory


async def test_transcript_chunks_go_and_report_chunks_stay(store, memory):
    old = await meeting_ended(store, days_ago(15))
    recent = await meeting_ended(store, days_ago(2))
    report = Report(
        meeting_id=old.id,
        summary="Alex refunds the double-charged users.",
        tasks=[TaskDraft(id="task-1", meeting_id=old.id, title="Refund the users")],
    )
    await memory.index_meeting(TEAM.id, old.id, await store.transcript(old.id), report)
    await memory.index_meeting(TEAM.id, recent.id, await store.transcript(recent.id))

    await purge_transcripts(store, memory, now=NOW, retention_days=14)

    hits = await memory.search(TEAM.id, "refund", k=50)
    left = {(h.chunk.meeting_id, h.chunk.kind) for h in hits}
    assert (old.id, "transcript") not in left
    assert {(old.id, "summary"), (old.id, "task"), (recent.id, "transcript")} <= left


async def test_without_memory_the_segments_go_and_the_result_says_chunks_stayed(store, memory):
    old = await meeting_ended(store, days_ago(15))
    await memory.index_meeting(TEAM.id, old.id, await store.transcript(old.id))

    result = await purge_transcripts(store, None, now=NOW, retention_days=14)

    assert [m.id for m in result.deleted] == [old.id]
    assert not result.memory_cleared
    assert await store.transcript(old.id) == []
    kinds = {h.chunk.kind for h in await memory.search(TEAM.id, "refund", k=50)}
    assert kinds == {"transcript"}


# failures


class FlakyMemory:
    def __init__(self, inner: MeetingMemory, failing: str):
        self.inner = inner
        self.failing = failing

    async def delete_meeting_transcript(self, meeting_id: str) -> None:
        if meeting_id == self.failing:
            raise RuntimeError("memory database unreachable")
        await self.inner.delete_meeting_transcript(meeting_id)


async def test_one_failing_meeting_does_not_stop_the_rest_and_is_retried_next_run(store, memory):
    first = await meeting_ended(store, days_ago(20), title="First")
    broken = await meeting_ended(store, days_ago(19), title="Broken")
    last = await meeting_ended(store, days_ago(18), title="Last")

    result = await purge_transcripts(
        store, FlakyMemory(memory, broken.id), now=NOW, retention_days=14
    )

    assert {m.id for m in result.deleted} == {first.id, last.id}
    assert "memory database unreachable" in result.failed[broken.id]
    # nothing about the broken one was marked deleted, so the next run picks it up again
    assert len(await store.transcript(broken.id)) == 2
    assert (await store.meeting(broken.id)).transcript_deleted_at is None

    retry = await purge_transcripts(store, memory, now=NOW, retention_days=14)

    assert [m.id for m in retry.deleted] == [broken.id]


# settings


def test_retention_defaults_to_fourteen_days():
    assert Settings(_env_file=None).transcript_retention_days == 14


def test_retention_below_one_day_is_refused():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, transcript_retention_days=0)


# the database stores retention runs against


async def test_retention_needs_a_database():
    with pytest.raises(RetentionUnavailable, match="DATABASE_URL is not configured"):
        async with open_retention_stores(Settings(_env_file=None)):
            pass


async def test_retention_purges_postgres_and_its_memory_on_one_pool(pg_dsn):
    await migrate(pg_dsn)
    settings = Settings(_env_file=None, database_url=pg_dsn)
    started = NOW - timedelta(days=20)

    async with open_retention_stores(settings) as (store, memory):
        assert isinstance(store, PostgresStore)
        assert isinstance(memory, PgMemoryStore)
        assert memory.pool is store.db
        await store.create_team(TEAM.model_copy(update={"member_ids": []}))
        await store.upsert_person(ALEX, TEAM.id)
        meeting = await store.create_meeting(TEAM.id, "Old", ALEX.id, scheduled_start=started)
        await store.start_meeting(meeting.id, started)
        await store.add_segments(meeting.id, segments(meeting.id))
        await store.transition_status(
            meeting.id, {"live"}, "processing", at=started + timedelta(minutes=30)
        )
        await store.set_status(meeting.id, "needs_review")
        await MeetingMemory(MockEmbedder(dim=memory.dim), memory).index_meeting(
            TEAM.id, meeting.id, segments(meeting.id)
        )

        result = await purge_transcripts(store, memory, now=NOW, retention_days=14)

        assert [m.id for m in result.deleted] == [meeting.id]
        assert await store.transcript(meeting.id) == []
        assert (await store.meeting(meeting.id)).transcript_deleted_at == NOW
        async with memory.pool.connection() as conn:
            cursor = await conn.execute(
                "select count(*) from memory_chunks where meeting_id = %s", [meeting.id]
            )
            assert await cursor.fetchone() == (0,)
