"""Transcript retention: once a meeting is older than the retention period its final segments and
their memory chunks are deleted; the report, decisions and tasks stay."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from contracts import Meeting

from .config import Settings
from .db import open_pool
from .memory import PgMemoryStore
from .pg_store import PostgresStore
from .store import Store

PURGEABLE = frozenset({"needs_review", "pushed"})
"""A live or processing meeting is never purged, however long ago it last ended."""


class TranscriptMemory(Protocol):
    """MeetingMemory, or a MemoryStore directly: deleting chunks needs no embedder."""

    async def delete_meeting_transcript(self, meeting_id: str) -> None: ...


class RetentionUnavailable(Exception):
    """What retention needs is not configured."""


@dataclass
class PurgeResult:
    deleted: list[Meeting] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    """Meeting id -> error. Nothing of a failed meeting is marked deleted; the next run retries."""
    memory_cleared: bool = True
    """False when no memory was given, so transcript chunks of the deleted meetings remain."""


def retention_cutoff(now: datetime, retention_days: int) -> datetime:
    if retention_days < 1:
        raise ValueError(f"retention must be at least 1 day, not {retention_days}")
    return now - timedelta(days=retention_days)


async def transcripts_due(store: Store, *, now: datetime, retention_days: int) -> list[Meeting]:
    """Meetings, across all teams, whose transcript is past retention. Changes nothing."""
    cutoff = retention_cutoff(now, retention_days)
    found = await store.meetings_with_transcript_before(cutoff)
    return [m for m in found if m.status in PURGEABLE]


async def purge_transcripts(
    store: Store, memory: TranscriptMemory | None, *, now: datetime, retention_days: int
) -> PurgeResult:
    """Deletes the transcript and transcript chunks of every meeting past retention. Safe to run
    repeatedly: a meeting already purged is not found again and keeps its first deletion time.
    Chunks go before the segments, so a meeting that fails part-way is still due next run."""
    result = PurgeResult(memory_cleared=memory is not None)
    for meeting in await transcripts_due(store, now=now, retention_days=retention_days):
        try:
            if memory is not None:
                await memory.delete_meeting_transcript(meeting.id)
            result.deleted.append(await store.delete_transcript(meeting.id, now))
        except Exception as e:  # one meeting must not stop the rest
            result.failed[meeting.id] = f"{type(e).__name__}: {e}"
    return result


@asynccontextmanager
async def open_retention_stores(
    settings: Settings,
) -> AsyncIterator[tuple[Store, TranscriptMemory | None]]:
    """The Postgres store and its meeting memory, on one pool that closes on exit."""
    if not settings.database_url:
        raise RetentionUnavailable("DATABASE_URL is not configured")
    pool = await open_pool(settings.database_url, max_size=2)
    try:
        yield PostgresStore(pool), PgMemoryStore(pool)
    finally:
        await pool.close()
