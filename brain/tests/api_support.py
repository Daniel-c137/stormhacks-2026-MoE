"""A two-team world and an app wired to it, for HTTP tests of the brain's API.

Auth is overridden: each test client sends X-Test-User, so several clients can act as
different people at once. Supabase session resolution is its own slice. The store is the real
in-memory store and LiveKit tokens are really signed.
"""

import pytest
from fastapi import Header
from fastapi.testclient import TestClient

from brain.api.deps import app_settings, current_user, get_rooms, get_store
from brain.config import Settings
from brain.main import create_app
from brain.store import InMemoryStore
from contracts import Person, Team

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
PEOPLE = {p.id: p for p in [ALEX, SARAH, OUTSIDER, NOBODY]}


def user_from_test_header(x_test_user: str = Header()) -> Person:
    return PEOPLE[x_test_user]


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
def settings() -> Settings:
    return Settings(
        _env_file=None,
        livekit_url=LIVEKIT_URL,
        livekit_api_key=KEY,
        livekit_api_secret=SECRET,
        brain_internal_token=WORKER_TOKEN,
    )


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(teams=[TEAM, OTHER_TEAM], people=[ALEX, SARAH, OUTSIDER])


@pytest.fixture
def rooms() -> FakeRooms:
    return FakeRooms()


@pytest.fixture
def app(store, settings, rooms):
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_rooms] = lambda: rooms
    app.dependency_overrides[app_settings] = lambda: settings
    app.dependency_overrides[current_user] = user_from_test_header
    return app


@pytest.fixture
def client_as(app):
    """client_as(person) -> a TestClient whose requests are made as that person. Each client
    carries its own identity, so several can be held at once."""

    def make(person: Person) -> TestClient:
        return TestClient(app, headers={"X-Test-User": person.id})

    return make


@pytest.fixture
def worker(app) -> TestClient:
    """The realtime worker: no user session, only the shared internal token."""
    return TestClient(app, headers={"X-Internal-Token": WORKER_TOKEN})


def create(client: TestClient, title: str = "Standup") -> dict:
    response = client.post("/meetings", json={"title": title})
    assert response.status_code == 200, response.text
    return response.json()
