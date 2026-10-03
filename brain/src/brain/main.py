from fastapi import FastAPI

from contracts import get_identity

from .api import routers


def create_app() -> FastAPI:
    app = FastAPI(title=f"{get_identity().product_name} brain")
    for router in routers:
        app.include_router(router)
    return app


app = create_app()
