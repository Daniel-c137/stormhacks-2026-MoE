"""A two-team world and an app wired to it, for HTTP tests of the brain's API.

Auth is overridden; Supabase session resolution is its own slice. The store is the real
in-memory store, or with BRAIN_TEST_STORE=postgres a PostgresStore on a fresh pgserver database
with the migrations applied. LiveKit tokens are really signed.
"""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from brain.api.deps import current_user, get_settings, get_store
from brain.config import Settings
from brain.main import create_app
from brain.store import InMemoryStore, Store
from contracts import Person, Team

KEY = "test-key"
SECRET = "test-secret-that-is-long-enough-for-hs256"
LIVEKIT_URL = "wss://omniroom-test.livekit.cloud"
WORKER_TOKEN = "worker-shared-secret"

ALEX = Person(id="u-alex", name="Alex Chen", short="Alex", initials="AC")
SARAH = Person(id="u-sarah", name="Sarah Kim", short="Sarah", initials="SK")
OUTSIDER = Person(id="u-olga", name="Olga Petrova", short="Olga", initials="OP")
TEAM = Team(id="t-1", name="Checkout", member_ids=[ALEX.id, SARAH.id])
OTHER_TEAM = Team(id="t-2", name="Elsewhere", member_ids=[OUTSIDER.id])


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        livekit_url=LIVEKIT_URL,
        livekit_api_key=KEY,
        livekit_api_secret=SECRET,
        brain_internal_token=WORKER_TOKEN,
    )


@pytest.fixture
def store(request) -> Store:
    if os.environ.get("BRAIN_TEST_STORE") == "postgres":
        return asyncio.run(postgres_world(request.getfixturevalue("pg_dsn")))
    return InMemoryStore(teams=[TEAM, OTHER_TEAM], people=[ALEX, SARAH, OUTSIDER])


async def postgres_world(dsn: str) -> Store:
    """The same two teams in Postgres. Connects per call, so any test's event loop can use it."""
    from brain.db import migrate
    from brain.pg_store import PostgresStore

    await migrate(dsn)
    store = PostgresStore(dsn)
    for team in (TEAM, OTHER_TEAM):
        await store.create_team(team.model_copy(update={"member_ids": []}))
    for person, team in ((ALEX, TEAM), (SARAH, TEAM), (OUTSIDER, OTHER_TEAM)):
        await store.upsert_person(person, team.id)
    return store


@pytest.fixture
def app(store, settings):
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
def client_as(app):
    """client_as(person) -> a TestClient whose requests are made as that person."""

    def make(person: Person) -> TestClient:
        app.dependency_overrides[current_user] = lambda: person
        return TestClient(app)

    return make


@pytest.fixture
def worker(app) -> TestClient:
    """The realtime worker: no user session, only the shared internal token."""
    return TestClient(app, headers={"X-Internal-Token": WORKER_TOKEN})


def create(client: TestClient, title: str = "Standup") -> dict:
    response = client.post("/meetings", json={"title": title})
    assert response.status_code == 200, response.text
    return response.json()
