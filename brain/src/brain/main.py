import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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


SECRET_FIELDS = frozenset({"api_token", "password", "current_password", "new_password"})
HIDDEN = "(hidden)"


def without_secrets(value: object) -> object:
    """An echoed input with its passwords and tokens hidden, at any depth."""
    if isinstance(value, dict):
        return {
            key: HIDDEN if key in SECRET_FIELDS else without_secrets(inner)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [without_secrets(inner) for inner in value]
    return value


async def validation_failed(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's 422, except that a non-finite number in the echoed input (JSON bodies may say
    Infinity or NaN, but responses cannot) is shown as text instead of failing with a 500, and a
    password or API token in it is never sent back."""
    finite = {float: lambda x: x if math.isfinite(x) else str(x)}
    errors = []
    for error in exc.errors():
        secret = bool(error.get("loc")) and error["loc"][-1] in SECRET_FIELDS
        shown = HIDDEN if secret else without_secrets(error.get("input"))
        errors.append({**error, "input": shown} if "input" in error else error)
    return JSONResponse(
        status_code=422, content={"detail": jsonable_encoder(errors, custom_encoder=finite)}
    )


def create_app() -> FastAPI:
    app = FastAPI(title=f"{get_identity().product_name} brain", lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, validation_failed)
    app.state.pipeline_runner = PipelineRunner()
    for router in routers:
        app.include_router(router)
    return app


app = create_app()
