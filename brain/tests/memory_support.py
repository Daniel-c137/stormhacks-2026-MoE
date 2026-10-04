"""An open pool on a fresh database with the meeting memory migration applied."""

from collections.abc import AsyncIterator
from pathlib import Path

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "supabase"
    / "migrations"
    / "20261003000100_meeting_memory.sql"
)


@pytest.fixture
async def memory_pool(pg_dsn) -> AsyncIterator[AsyncConnectionPool]:
    """Use from anyio tests."""
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(MIGRATION.read_text())
    pool = AsyncConnectionPool(pg_dsn, min_size=1, max_size=4, open=False)
    await pool.open(wait=True)
    yield pool
    await pool.close()
