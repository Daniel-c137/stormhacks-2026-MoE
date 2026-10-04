"""The migration runner and the connection pool, against a real Postgres (pgserver)."""

from pathlib import Path

import anyio
import psycopg
import pytest

from brain.db import MIGRATIONS, migrate, open_pool

pytestmark = pytest.mark.anyio


def write(directory: Path, name: str, sql: str) -> None:
    (directory / name).write_text(sql)


def query(dsn: str, sql: str) -> list[tuple]:
    with psycopg.connect(dsn) as conn:
        return conn.execute(sql).fetchall()


def applied(dsn: str) -> list[str]:
    return [row[0] for row in query(dsn, "select version from schema_migrations order by version")]


async def test_each_migration_is_applied_once_in_filename_order(pg_dsn, tmp_path):
    write(tmp_path, "20260102000000_fill.sql", "insert into notes values ('second');")
    write(tmp_path, "20260101000000_create.sql", "create table notes (text text); ")

    assert await migrate(pg_dsn, tmp_path) == ["20260101000000_create", "20260102000000_fill"]
    assert await migrate(pg_dsn, tmp_path) == []

    write(tmp_path, "20260103000000_more.sql", "insert into notes values ('third');")
    assert await migrate(pg_dsn, tmp_path) == ["20260103000000_more"]

    assert query(pg_dsn, "select text from notes") == [("second",), ("third",)]
    assert applied(pg_dsn) == [
        "20260101000000_create",
        "20260102000000_fill",
        "20260103000000_more",
    ]


async def test_a_failing_migration_leaves_no_trace_and_stops_the_run(pg_dsn, tmp_path):
    write(tmp_path, "1_good.sql", "create table good (id int);")
    write(tmp_path, "2_bad.sql", "create table half (id int); select no_such_function();")
    write(tmp_path, "3_later.sql", "create table later (id int);")

    with pytest.raises(psycopg.Error):
        await migrate(pg_dsn, tmp_path)

    assert applied(pg_dsn) == ["1_good"]
    tables = {
        r[0] for r in query(pg_dsn, "select tablename from pg_tables where schemaname = 'public'")
    }
    assert "good" in tables
    assert not {"half", "later"} & tables


async def test_overlapping_runs_apply_each_migration_once(pg_dsn, tmp_path):
    write(tmp_path, "1_create.sql", "create table counter (n int); insert into counter values (1);")
    write(tmp_path, "2_bump.sql", "update counter set n = n + 1;")

    async with anyio.create_task_group() as tg:
        for _ in range(4):
            tg.start_soon(migrate, pg_dsn, tmp_path)

    assert query(pg_dsn, "select n from counter") == [(2,)]
    assert applied(pg_dsn) == ["1_create", "2_bump"]


async def test_migrate_also_takes_an_open_pool(pg_dsn, tmp_path):
    write(tmp_path, "1_create.sql", "create table notes (text text);")
    pool = await open_pool(pg_dsn)
    try:
        assert await migrate(pool, tmp_path) == ["1_create"]
        async with pool.connection() as conn:
            assert await (await conn.execute("select count(*) from notes")).fetchone() == (0,)
    finally:
        await pool.close()


async def test_the_repos_migrations_build_the_core_schema_and_memory(pg_dsn):
    names = await migrate(pg_dsn)

    assert names == sorted(p.stem for p in MIGRATIONS.glob("*.sql"))
    assert names[0].endswith("_core")
    tables = {
        r[0] for r in query(pg_dsn, "select tablename from pg_tables where schemaname = 'public'")
    }
    assert {
        "teams",
        "people",
        "memberships",
        "team_settings",
        "meetings",
        "transcript_segments",
        "public_chat",
        "agendas",
        "agenda_items",
        "reports",
        "report_progress",
        "task_drafts",
        "decisions",
        "memory_chunks",
        "schema_migrations",
    } <= tables
    assert not [t for t in tables if "private" in t]


async def test_no_core_table_is_readable_through_supabases_client_keys(pg_dsn):
    """Row level security with no policies: only the brain's own connection sees rows."""
    await migrate(pg_dsn)

    exposed = query(
        pg_dsn,
        "select relname from pg_class c join pg_namespace n on n.oid = c.relnamespace "
        "where n.nspname = 'public' and c.relkind = 'r' and not c.relrowsecurity",
    )

    assert exposed == []
