import secrets
from functools import cache
from typing import NoReturn

from fastapi import Depends, Header, HTTPException

from contracts import Meeting, Person, Team

from ..config import Settings
from ..store import NotFound, Store


def not_implemented() -> NoReturn:
    raise HTTPException(status_code=501, detail="Not implemented")


@cache
def get_settings() -> Settings:
    return Settings()


async def get_store() -> Store:
    """The Supabase store. Until it exists, tests and local runs override this dependency."""
    not_implemented()


async def current_user(authorization: str = Header()) -> Person:
    """Resolve the Supabase session; every query is scoped to this user's team."""
    not_implemented()


async def require_internal(
    x_internal_token: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Guard for realtime -> brain calls (BRAIN_INTERNAL_TOKEN)."""
    expected = settings.brain_internal_token
    if not expected:
        raise HTTPException(status_code=503, detail="BRAIN_INTERNAL_TOKEN is not configured")
    if not x_internal_token or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="Invalid internal token")


async def user_team(store: Store, user: Person) -> Team:
    try:
        return await store.team_for_user(user.id)
    except NotFound:
        raise HTTPException(status_code=403, detail="You are not on a team") from None


async def team_meeting(store: Store, user: Person, meeting_id: str) -> Meeting:
    """The meeting if it belongs to the user's team. Other teams' meetings are a plain 404."""
    team = await user_team(store, user)
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        meeting = None
    if meeting is None or meeting.team_id != team.id:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting
