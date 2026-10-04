"""Which store the app uses: Postgres when DATABASE_URL is set, otherwise a clear 503."""

import asyncio

from api_support import ALEX, OTHER_TEAM, OUTSIDER, SARAH, TEAM
from fastapi.testclient import TestClient

from brain.api.deps import current_user, get_settings
from brain.config import Settings
from brain.db import migrate
from brain.main import create_app
from brain.pg_store import PostgresStore


def app_with(settings: Settings):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[current_user] = lambda: ALEX
    return app


def test_without_database_url_the_store_is_unavailable_not_in_memory():
    app = app_with(Settings(_env_file=None))

    with TestClient(app) as client:
        response = client.get("/meetings")

    assert response.status_code == 503
    assert response.json()["detail"] == "DATABASE_URL is not configured"


def test_database_url_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://brain@localhost/brain")

    assert Settings(_env_file=None).database_url == "postgresql://brain@localhost/brain"


def test_with_database_url_the_app_saves_to_postgres_through_a_pool_it_closes(pg_dsn):
    async def seed() -> PostgresStore:
        await migrate(pg_dsn)
        store = PostgresStore(pg_dsn)
        for team in (TEAM, OTHER_TEAM):
            await store.create_team(team.model_copy(update={"member_ids": []}))
        for person, team in ((ALEX, TEAM), (SARAH, TEAM), (OUTSIDER, OTHER_TEAM)):
            await store.upsert_person(person, team.id)
        return store

    store = asyncio.run(seed())
    app = app_with(Settings(_env_file=None, database_url=pg_dsn))

    with TestClient(app) as client:
        created = client.post("/meetings", json={"title": "Standup"})
        assert created.status_code == 200, created.text
        listed = client.get("/meetings").json()
        pool = app.state.db_pool

    assert pool.closed
    meeting = asyncio.run(store.meeting(created.json()["id"]))
    assert (meeting.team_id, meeting.host_id, meeting.title) == (TEAM.id, ALEX.id, "Standup")
    assert [m["id"] for m in listed] == [meeting.id]
