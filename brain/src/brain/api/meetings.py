"""Live meeting lifecycle and in-meeting questions (board -> brain)."""

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
from ..livekit_tokens import participant_token
from ..store import NotFound, Store
from .deps import app_settings, current_user, get_store, not_implemented

router = APIRouter(prefix="/meetings", tags=["meetings"])


async def _team(store: Store, user: Person) -> Team:
    try:
        return await store.team_for_user(user.id)
    except NotFound:
        raise HTTPException(status_code=403, detail="You are not on a team") from None


async def _team_meeting(store: Store, user: Person, meeting_id: str) -> Meeting:
    """The meeting if it belongs to the user's team. Other teams' meetings are a plain 404."""
    team = await _team(store, user)
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        meeting = None
    if meeting is None or meeting.team_id != team.id:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.post("")
async def create_meeting(
    body: CreateMeetingRequest,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> Meeting:
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title is required")
    team = await _team(store, user)
    return await store.create_meeting(team.id, title, host_id=user.id)


@router.get("")
async def list_meetings(
    user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> list[Meeting]:
    team = await _team(store, user)
    return await store.meetings(team.id)


@router.get("/{meeting_id}")
async def get_meeting(
    meeting_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Meeting:
    return await _team_meeting(store, user, meeting_id)


@router.post("/join/{code}")
async def join_meeting(
    code: str,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    settings: Settings = Depends(app_settings),
) -> JoinMeetingResponse:
    """Team members with a valid link join directly and get a LiveKit token."""
    try:
        meeting = await store.meeting_by_code(code)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    team = await _team(store, user)
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
    )
    return JoinMeetingResponse(meeting=meeting, livekit_url=settings.livekit_url, token=token)


@router.post("/{meeting_id}/end")
async def end_meeting(
    meeting_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Meeting:
    """Host only. Starts the post-meeting pipeline."""
    meeting = await _team_meeting(store, user, meeting_id)
    if meeting.host_id != user.id:
        raise HTTPException(status_code=403, detail="Only the host can end the meeting")
    if meeting.status != "live":
        return meeting
    # The post-meeting pipeline hooks in here once it exists; status alone gates it to run once.
    return await store.set_status(meeting.id, "processing")


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
