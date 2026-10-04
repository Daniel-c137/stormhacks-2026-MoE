"""Postgres connections and the migration runner.

`migrate` applies db/migrations to our Postgres 16 + pgvector, in every environment: `brain
migrate` from the command line, and directly in tests.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from .config import REPO_ROOT

MIGRATIONS = REPO_ROOT / "db" / "migrations"

# Any constant works; it only has to be the same for every run against one database.
_MIGRATION_LOCK = 0x6272_6169_6E  # "brain"

# prepare_threshold=None: transaction-mode poolers such as PgBouncer do not support prepared
# statements, so the brain never relies on them.
CONNECT_KWARGS = {"prepare_threshold": None}


async def open_pool(dsn: str, *, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    """An open pool; close it with `await pool.close()`."""
    pool = AsyncConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs=CONNECT_KWARGS,
        check=AsyncConnectionPool.check_connection,
        open=False,
    )
    await pool.open(wait=True)
    return pool


@asynccontextmanager
async def connection(db: AsyncConnectionPool | str) -> AsyncIterator[AsyncConnection]:
    """A connection from the pool, or a new one to the DSN. Commits on success."""
    if isinstance(db, str):
        async with await AsyncConnection.connect(db, **CONNECT_KWARGS) as conn:
            yield conn
    else:
        async with db.connection() as conn:
            yield conn


async def migrate(db: AsyncConnectionPool | str, directory: Path = MIGRATIONS) -> list[str]:
    """Applies the directory's *.sql files not applied yet, in filename order, each in its own
    transaction, and returns their names (without .sql). A failing file is rolled back and stops
    the run. Overlapping runs wait for each other, so each file is applied once."""
    done: list[str] = []
    async with connection(db) as conn:
        await conn.commit()  # each file below gets its own transaction
        for path in sorted(directory.glob("*.sql")):
            async with conn.transaction():
                await conn.execute("select pg_advisory_xact_lock(%s)", [_MIGRATION_LOCK])
                await conn.execute(
                    "create table if not exists schema_migrations ("
                    " version text primary key,"
                    " applied_at timestamptz not null default now())"
                )
                await conn.execute("alter table schema_migrations enable row level security")
                cursor = await conn.execute(
                    "select 1 from schema_migrations where version = %s", [path.stem]
                )
                if await cursor.fetchone():
                    continue
                await conn.execute(path.read_text())
                await conn.execute(
                    "insert into schema_migrations (version) values (%s)", [path.stem]
                )
            done.append(path.stem)
    return done
