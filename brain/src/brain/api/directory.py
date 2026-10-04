"""The caller's workspace directory, for picking invitees (board -> brain)."""

from fastapi import APIRouter, Depends, Query

from contracts import Person

from ..store import Store
from .deps import current_user, get_store, user_team

router = APIRouter(tags=["directory"])


@router.get("/directory")
async def search_directory(
    q: str = "",
    limit: int = Query(default=20, ge=1, le=100),
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> list[Person]:
    """Members of the caller's workspace matching q by name, email or title; blank lists all."""
    team = await user_team(store, user)
    return await store.search_members(team.id, q, limit)
