"""A two-team world and an app wired to it, for HTTP tests of the brain's API.

Auth is overridden: each client_as client sends X-Test-User, so several clients can act as
different people at once; test_auth.py covers real Supabase sessions. The store is the real
in-memory store, or with BRAIN_TEST_STORE=postgres a PostgresStore on a fresh pgserver database
with the migrations applied. LiveKit tokens are really signed.
"""

import asyncio
import os

import pytest
from fastapi import Header, Request
from fastapi.testclient import TestClient

from brain.api.deps import current_user, get_rooms, get_settings, get_store
from brain.config import Settings
from brain.main import create_app
from brain.store import InMemoryStore, Store
from contracts import AGENT_PARTICIPANT_ID, Person, Team

KEY = "test-key"
SECRET = "test-secret-that-is-long-enough-for-hs256"
LIVEKIT_URL = "wss://omniroom-test.livekit.cloud"
WORKER_TOKEN = "worker-shared-secret-0123456789abcdef"

ALEX = Person(id="u-alex", name="Alex Chen", short="Alex", initials="AC")
SARAH = Person(id="u-sarah", name="Sarah Kim", short="Sarah", initials="SK")
OUTSIDER = Person(id="u-olga", name="Olga Petrova", short="Olga", initials="OP")
NOBODY = Person(id="u-nobody", name="No Team", short="No", initials="NT")  # on no team
TEAM = Team(id="t-1", name="Checkout", member_ids=[ALEX.id, SARAH.id])
OTHER_TEAM = Team(id="t-2", name="Elsewhere", member_ids=[OUTSIDER.id])


def user_from_test_header(request: Request, x_test_user: str = Header()) -> Person:
    """The person the requesting client_as client was made for."""
    return request.app.state.test_people[x_test_user]


class FakeRooms:
    """Records LiveKit rooms closed; fail=True makes closing raise like an unreachable LiveKit."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.closed: list[str] = []

    async def close(self, room: str) -> None:
        if self.fail:
            raise RuntimeError("LiveKit unreachable")
        self.closed.append(room)


@pytest.fixture
def rooms() -> FakeRooms:
    return FakeRooms()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        livekit_url=LIVEKIT_URL,
        livekit_api_key=KEY,
        livekit_api_secret=SECRET,
        brain_internal_token=WORKER_TOKEN,
        gemini_api_key=None,  # never real Gemini; tests that need a model override get_llm*
        # A sync TestClient closes its event loop after each request, so a write-up it starts
        # is always still settling then and is cut off; ending a meeting stays at processing.
        # test_pipeline_api.py runs write-ups to the end on its own loop with no settle time.
        pipeline_settle_seconds=3600,
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
def app(store, settings, rooms):
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rooms] = lambda: rooms
    return app


@pytest.fixture
def client_as(app):
    """client_as(person) -> a TestClient whose requests are made as that person. Each client
    carries its own identity, so several can be held at once."""

    app.state.test_people = {}
    app.dependency_overrides[current_user] = user_from_test_header

    def make(person: Person) -> TestClient:
        app.state.test_people[person.id] = person
        return TestClient(app, headers={"X-Test-User": person.id})

    return make


@pytest.fixture
def worker(app) -> TestClient:
    """The realtime worker: no user session, only the shared internal token."""
    return TestClient(app, headers={"X-Internal-Token": WORKER_TOKEN})


async def speakers_join(app, meeting_id: str, segments: list[dict]) -> None:
    """Everyone speaking in these segments joins the meeting first, as through the link: the
    brain only takes segments from the meeting's participants (or the agent)."""
    store = app.dependency_overrides[get_store]()
    for speaker_id in sorted({s["speaker_id"] for s in segments} - {AGENT_PARTICIPANT_ID}):
        await store.add_participant(meeting_id, speaker_id)


def create(client: TestClient, title: str = "Standup") -> dict:
    response = client.post("/meetings", json={"title": title})
    assert response.status_code == 200, response.text
    return response.json()
