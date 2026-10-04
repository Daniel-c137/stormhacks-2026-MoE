"""Real Postgres 16 with pgvector for tests, from the `pgserver` package: no Docker, no network.

One server per test session; every test that asks for `pg_dsn` gets a fresh, empty database
that is dropped afterwards. Tests apply the migrations they need themselves.
"""

import shutil
import tempfile
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql


@pytest.fixture(scope="session")
def pg_server():
    """The session's embedded Postgres server (a `pgserver.PostgresServer`)."""
    pgserver = pytest.importorskip("pgserver")
    # A short directory: the server's unix socket lives in it and socket paths are length-capped.
    data_dir = tempfile.mkdtemp(prefix="pg")
    server = pgserver.get_server(data_dir, cleanup_mode="delete")
    yield server
    server.cleanup()
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def pg_dsn(pg_server) -> Iterator[str]:
    """The DSN of a fresh database on the session's server, dropped after the test."""
    name = f"test_{uuid4().hex}"
    with psycopg.connect(pg_server.get_uri(), autocommit=True) as admin:
        admin.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
    yield pg_server.get_uri(name)
    with psycopg.connect(pg_server.get_uri(), autocommit=True) as admin:
        admin.execute(sql.SQL("drop database {} with (force)").format(sql.Identifier(name)))
