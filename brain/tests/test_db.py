"""The migration runner and the connection pool, against a real Postgres (pgserver)."""

from pathlib import Path

import anyio
import psycopg
import pytest

from brain.config import REPO_ROOT
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
    assert MIGRATIONS == REPO_ROOT / "db" / "migrations"
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
        "report_audio",
        "memory_chunks",
        "logins",
        "schema_migrations",
    } <= tables
    assert not [t for t in tables if "private" in t]


async def test_every_table_has_row_level_security_on(pg_dsn):
    """Row level security with no policies: only the role that owns the tables, the brain's own,
    sees rows; any other role on the server reads nothing."""
    await migrate(pg_dsn)

    exposed = query(
        pg_dsn,
        "select relname from pg_class c join pg_namespace n on n.oid = c.relnamespace "
        "where n.nspname = 'public' and c.relkind = 'r' and not c.relrowsecurity",
    )

    assert exposed == []


async def test_a_meetings_report_audio_goes_with_the_meeting(pg_dsn):
    await migrate(pg_dsn)
    with psycopg.connect(pg_dsn) as conn:
        conn.execute("insert into teams (id, name) values ('t', 'Checkout')")
        conn.execute(
            "insert into meetings (id, team_id, title, status, code, host_id)"
            " values ('m', 't', 'Standup', 'needs_review', 'code', 'u')"
        )
        conn.execute(
            "insert into report_audio (meeting_id, key, content_type, data)"
            " values ('m', 'k', 'audio/mpeg', %s)",
            [b"ID3"],
        )
        conn.execute("delete from meetings where id = 'm'")

    assert query(pg_dsn, "select meeting_id from report_audio") == []


async def test_a_persons_login_goes_with_them_and_emails_are_unique_ignoring_case(pg_dsn):
    await migrate(pg_dsn)
    with psycopg.connect(pg_dsn) as conn:
        for pid in ("a", "b"):
            conn.execute(
                "insert into people (id, name, short, initials) values (%s, 'N', 'N', 'N')", [pid]
            )
        conn.execute(
            "insert into logins (person_id, email, password_hash) values ('a', 'A@x.dev', 'h')"
        )
        with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
            conn.execute(
                "insert into logins (person_id, email, password_hash) values ('b', 'a@X.dev', 'h')"
            )
        conn.execute("delete from people where id = 'a'")

    assert query(pg_dsn, "select person_id from logins") == []


CODE_CONNECTORS = "20261004001000_code_connectors.sql"


async def test_the_single_repository_moves_into_the_list_and_gitlab_starts_empty(pg_dsn, tmp_path):
    """Settings saved before several repositories keep their repository, ref and index state;
    an unset repository becomes an empty list; the Jira site loses its scheme."""
    for path in sorted(MIGRATIONS.glob("*.sql")):
        if path.name < CODE_CONNECTORS:
            write(tmp_path, path.name, path.read_text())
    await migrate(pg_dsn, tmp_path)
    with psycopg.connect(pg_dsn) as conn:
        for team, github, jira in (
            (
                "t1",
                '{"repo": "acme/checkout", "ref": "main", "connected": true,'
                ' "indexed_at": "2026-10-01T12:00:00Z", "files": 420}',
                '{"site": "https://acme.atlassian.net/", "project": "DS", "connected": false}',
            ),
            ("t2", '{"repo": null, "ref": null, "connected": false}', '{"site": null}'),
            ("t3", '{"repo": "  ", "connected": false}', '{"site": "acme.atlassian.net"}'),
        ):
            conn.execute("insert into teams (id, name) values (%s, %s)", [team, team])
            conn.execute(
                "insert into team_settings (team_id, github, jira, sensitivity,"
                " interrupt_minutes, who_can_allow) values (%s, %s, %s, 'balanced', 5, 'everyone')",
                [team, github, jira],
            )

    write(tmp_path, CODE_CONNECTORS, (MIGRATIONS / CODE_CONNECTORS).read_text())
    assert await migrate(pg_dsn, tmp_path) == [CODE_CONNECTORS.removesuffix(".sql")]

    rows = dict(
        (team, (github, gitlab, jira))
        for team, github, gitlab, jira in query(
            pg_dsn, "select team_id, github, gitlab, jira from team_settings order by team_id"
        )
    )
    assert rows["t1"][0] == {
        "repos": [
            {
                "path": "acme/checkout",
                "ref": "main",
                "connected": True,
                "indexed_at": "2026-10-01T12:00:00Z",
                "files": 420,
            }
        ]
    }
    assert rows["t1"][1] == {"projects": []}
    assert rows["t1"][2]["site"] == "acme.atlassian.net"
    assert rows["t2"][0] == {"repos": []} and rows["t2"][2]["site"] is None
    assert rows["t3"][0] == {"repos": []} and rows["t3"][2]["site"] == "acme.atlassian.net"
