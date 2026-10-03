"""Team, members and workspace settings (board -> brain)."""

from fastapi import APIRouter, Depends

from contracts import Person, Team, TeamSettings, Voice

from .deps import current_user, not_implemented

router = APIRouter(tags=["team"])


@router.get("/team")
async def get_team(user: Person = Depends(current_user)) -> Team:
    not_implemented()


@router.get("/team/members")
async def list_members(user: Person = Depends(current_user)) -> list[Person]:
    not_implemented()


@router.get("/settings")
async def get_settings(user: Person = Depends(current_user)) -> TeamSettings:
    not_implemented()


@router.put("/settings")
async def update_settings(body: TeamSettings, user: Person = Depends(current_user)) -> TeamSettings:
    not_implemented()


@router.get("/voices")
async def list_voices(user: Person = Depends(current_user)) -> list[Voice]:
    not_implemented()
