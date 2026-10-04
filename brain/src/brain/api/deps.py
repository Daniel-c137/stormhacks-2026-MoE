import secrets
from functools import cache
from typing import NoReturn

from fastapi import Depends, Header, HTTPException

from contracts import Meeting, Person, Team

from ..config import Settings
from ..livekit_rooms import Rooms, rooms_from_settings
from ..store import NotFound, Store


def not_implemented() -> NoReturn:
    raise HTTPException(status_code=501, detail="Not implemented")


@cache
def app_settings() -> Settings:
    return Settings()


async def get_store() -> Store:
    """The Supabase store. Until it exists, tests and local runs override this dependency."""
    not_implemented()


def get_rooms(settings: Settings = Depends(app_settings)) -> Rooms:
    return rooms_from_settings(settings)


async def current_user(authorization: str = Header()) -> Person:
    """Resolve the Supabase session; every query is scoped to this user's team."""
    not_implemented()


MIN_INTERNAL_TOKEN = 32


async def require_internal(
    x_internal_token: str | None = Header(default=None),
    settings: Settings = Depends(app_settings),
) -> None:
    """Guard for realtime -> brain calls (BRAIN_INTERNAL_TOKEN). A short token is treated as
    unset: /internal shares the board's port, so the token is the only thing protecting it."""
    expected = settings.brain_internal_token
    if not expected or len(expected) < MIN_INTERNAL_TOKEN:
        raise HTTPException(
            status_code=503,
            detail=f"BRAIN_INTERNAL_TOKEN must be set to at least {MIN_INTERNAL_TOKEN} characters",
        )
    # Bytes: compare_digest raises on non-ASCII str, which would turn a bad header into a 500.
    if not x_internal_token or not secrets.compare_digest(
        x_internal_token.encode(), expected.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid internal token")


# Team scoping is a dependency, not a helper, so a route can't forget it.


async def user_team(
    user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Team:
    try:
        return await store.team_for_user(user.id)
    except NotFound:
        raise HTTPException(status_code=403, detail="You are not on a team") from None


async def member_meeting(
    meeting_id: str, team: Team = Depends(user_team), store: Store = Depends(get_store)
) -> Meeting:
    """The path's meeting if it is the user's team's. Other teams' meetings are a plain 404."""
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        meeting = None
    if meeting is None or meeting.team_id != team.id:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting
