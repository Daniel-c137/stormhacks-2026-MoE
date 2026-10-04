from functools import cache
from typing import NoReturn

from fastapi import Header, HTTPException

from contracts import Person

from ..config import Settings
from ..store import Store


def not_implemented() -> NoReturn:
    raise HTTPException(status_code=501, detail="Not implemented")


@cache
def app_settings() -> Settings:
    return Settings()


async def get_store() -> Store:
    """The Supabase store. Until it exists, tests and local runs override this dependency."""
    not_implemented()


async def current_user(authorization: str = Header()) -> Person:
    """Resolve the Supabase session; every query is scoped to this user's team."""
    not_implemented()


async def require_internal(x_internal_token: str = Header()) -> None:
    """Guard for realtime -> brain calls (BRAIN_INTERNAL_TOKEN)."""
    not_implemented()
