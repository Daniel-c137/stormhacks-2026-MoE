from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from contracts import get_identity

from .agent.pipeline import PipelineRunner
from .api import routers
from .api.deps import get_settings
from .db import open_pool
from .pg_store import PostgresStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """With DATABASE_URL, one pool for the app's life: app.state.db_pool, and the Postgres store
    on it as app.state.store. Without it both stay None and the store routes answer 503.
    Write-ups still running at shutdown are stopped first, so they can record the interruption."""
    settings = app.dependency_overrides.get(get_settings, get_settings)()
    runner: PipelineRunner = app.state.pipeline_runner
    app.state.db_pool = app.state.store = None
    if not settings.database_url:
        try:
            yield
        finally:
            await runner.shutdown()
        return
    pool = await open_pool(settings.database_url)
    app.state.db_pool, app.state.store = pool, PostgresStore(pool)
    try:
        yield
    finally:
        await runner.shutdown()
        app.state.db_pool = app.state.store = None
        await pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title=f"{get_identity().product_name} brain", lifespan=lifespan)
    app.state.pipeline_runner = PipelineRunner()
    for router in routers:
        app.include_router(router)
    return app


app = create_app()
