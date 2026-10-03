"""Live meeting lifecycle and in-meeting questions (board -> brain)."""

from fastapi import APIRouter, Depends

from contracts import (
    Agenda,
    Answer,
    AskRequest,
    CreateMeetingRequest,
    JoinMeetingResponse,
    Meeting,
    Person,
)

from .deps import current_user, not_implemented

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.post("")
async def create_meeting(
    body: CreateMeetingRequest, user: Person = Depends(current_user)
) -> Meeting:
    not_implemented()


@router.get("")
async def list_meetings(user: Person = Depends(current_user)) -> list[Meeting]:
    not_implemented()


@router.get("/{meeting_id}")
async def get_meeting(meeting_id: str, user: Person = Depends(current_user)) -> Meeting:
    not_implemented()


@router.post("/join/{code}")
async def join_meeting(code: str, user: Person = Depends(current_user)) -> JoinMeetingResponse:
    """Team members with a valid link join directly and get a LiveKit token."""
    not_implemented()


@router.post("/{meeting_id}/end")
async def end_meeting(meeting_id: str, user: Person = Depends(current_user)) -> Meeting:
    """Host only. Starts the post-meeting pipeline."""
    not_implemented()


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
