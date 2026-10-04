"""Live meeting lifecycle and in-meeting questions (board -> brain)."""

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException

from contracts import (
    Agenda,
    Answer,
    AskRequest,
    CreateMeetingRequest,
    JoinMeetingResponse,
    Meeting,
    Person,
    Team,
)

from ..config import Settings
from ..livekit_rooms import Rooms
from ..livekit_tokens import participant_token
from ..store import NotFound, Store
from .deps import (
    app_settings,
    current_user,
    get_rooms,
    get_store,
    member_meeting,
    not_implemented,
    user_team,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/meetings", tags=["meetings"])


MAX_TITLE = 200


@router.post("")
async def create_meeting(
    body: CreateMeetingRequest,
    user: Person = Depends(current_user),
    team: Team = Depends(user_team),
    store: Store = Depends(get_store),
) -> Meeting:
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title is required")
    if len(title) > MAX_TITLE:
        raise HTTPException(status_code=422, detail=f"Title is over {MAX_TITLE} characters")
    return await store.create_meeting(team.id, title, host_id=user.id)


@router.get("")
async def list_meetings(
    team: Team = Depends(user_team), store: Store = Depends(get_store)
) -> list[Meeting]:
    """Newest first."""
    return await store.meetings(team.id)


@router.get("/{meeting_id}")
async def get_meeting(meeting: Meeting = Depends(member_meeting)) -> Meeting:
    return meeting


@router.post("/join/{code}")
async def join_meeting(
    code: str,
    user: Person = Depends(current_user),
    team: Team = Depends(user_team),
    store: Store = Depends(get_store),
    settings: Settings = Depends(app_settings),
) -> JoinMeetingResponse:
    """Team members with a valid link join directly and get a short-lived LiveKit token."""
    try:
        meeting = await store.meeting_by_code(code)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    if meeting.team_id != team.id:
        raise HTTPException(status_code=403, detail="Only team members can join this meeting")
    if meeting.status != "live":
        raise HTTPException(status_code=409, detail="This meeting has ended")
    if not (settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret):
        raise HTTPException(status_code=503, detail="LiveKit is not configured")

    meeting = await store.add_participant(meeting.id, user.id)
    token = participant_token(
        meeting,
        user,
        is_host=meeting.host_id == user.id,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
        ttl=timedelta(seconds=settings.livekit_token_ttl_seconds),
    )
    return JoinMeetingResponse(meeting=meeting, livekit_url=settings.livekit_url, token=token)


@router.post("/{meeting_id}/end")
async def end_meeting(
    meeting: Meeting = Depends(member_meeting),
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    rooms: Rooms = Depends(get_rooms),
) -> Meeting:
    """Host only. Closes the LiveKit room and starts the post-meeting pipeline, exactly once."""
    if meeting.host_id != user.id:
        raise HTTPException(status_code=403, detail="Only the host can end the meeting")
    ended = await store.end_meeting(meeting.id)
    if ended is None:  # someone else's end already won
        return await store.meeting(meeting.id)
    try:
        await rooms.close(meeting.id)
    except Exception:
        log.warning("Could not close LiveKit room %s", meeting.id, exc_info=True)
    # The post-meeting pipeline starts here, only for the caller whose end won.
    return ended


@router.post("/{meeting_id}/ask")
async def ask_in_meeting(
    meeting_id: str, body: AskRequest, user: Person = Depends(current_user)
) -> Answer:
    """Typed question. Private answers return only to the asker; public ones also reach the room."""
    not_implemented()


@router.get("/{meeting_id}/agenda")
async def get_agenda(meeting_id: str, user: Person = Depends(current_user)) -> Agenda:
    not_implemented()


@router.post("/{meeting_id}/agenda")
async def generate_agenda(meeting_id: str, user: Person = Depends(current_user)) -> Agenda:
    """Build the agenda from previous meetings and unfinished GitHub/Jira work."""
    not_implemented()
