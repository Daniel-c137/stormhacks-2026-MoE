"""Meeting lifecycle (schedule, invite, join, end) and in-meeting questions (board -> brain)."""

from collections.abc import Iterable
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException

from contracts import (
    Answer,
    AskRequest,
    CreateMeetingRequest,
    InviteRequest,
    JoinMeetingResponse,
    Meeting,
    Person,
)

from ..config import Settings
from ..livekit_tokens import participant_token
from ..store import Conflict, NotFound, Store
from .deps import (
    current_user,
    get_settings,
    get_store,
    not_implemented,
    team_meeting,
    user_team,
)

router = APIRouter(prefix="/meetings", tags=["meetings"])

MAX_DURATION_MIN = 24 * 60


async def team_invitees(store: Store, team_id: str, host_id: str, ids: Iterable[str]) -> list[str]:
    """The ids in order without duplicates or the host; 422 naming any who are not on the team."""
    wanted = [i for i in dict.fromkeys(ids) if i != host_id]
    members = {p.id for p in await store.members(team_id)}
    unknown = [i for i in wanted if i not in members]
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Not members of your workspace: {', '.join(unknown)}"
        )
    return wanted


async def end_live_meeting(store: Store, meeting_id: str) -> tuple[Meeting, bool]:
    """live -> processing as one compare-and-set. True only for the call that made the move, so
    of overlapping ends exactly one starts the post-meeting pipeline."""
    try:
        return await store.transition_status(meeting_id, {"live"}, "processing"), True
    except Conflict:
        return await store.meeting(meeting_id), False


async def host_meeting(store: Store, user: Person, meeting_id: str) -> Meeting:
    meeting = await team_meeting(store, user, meeting_id)
    if meeting.host_id != user.id:
        raise HTTPException(status_code=403, detail="Only the host can change invitees")
    if meeting.status not in ("scheduled", "live"):
        raise HTTPException(status_code=409, detail="This meeting has ended")
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
    start = body.scheduled_start
    if start is not None and start.utcoffset() is None:
        raise HTTPException(status_code=422, detail="scheduled_start needs a timezone")
    if body.duration_min is not None and not 0 < body.duration_min <= MAX_DURATION_MIN:
        raise HTTPException(
            status_code=422, detail=f"duration_min must be between 1 and {MAX_DURATION_MIN}"
        )
    team = await user_team(store, user)
    invitees = await team_invitees(store, team.id, user.id, body.invitee_ids)
    return await store.create_meeting(
        team.id,
        title,
        host_id=user.id,
        scheduled_start=start,
        duration_min=body.duration_min,
        invitee_ids=invitees,
    )


@router.get("")
async def list_meetings(
    user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> list[Meeting]:
    team = await user_team(store, user)
    return await store.meetings(team.id)


@router.get("/{meeting_id}")
async def get_meeting(
    meeting_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Meeting:
    return await team_meeting(store, user, meeting_id)


@router.post("/join/{code}")
async def join_meeting(
    code: str,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> JoinMeetingResponse:
    """Team members with a valid link join directly and get a LiveKit token. A scheduled meeting
    starts when its host joins; until then everyone else is told when it is scheduled."""
    try:
        meeting = await store.meeting_by_code(code)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    team = await user_team(store, user)
    if meeting.team_id != team.id:
        raise HTTPException(status_code=403, detail="Only team members can join this meeting")
    if meeting.status == "scheduled":
        if meeting.host_id != user.id:
            when = meeting.scheduled_start.isoformat() if meeting.scheduled_start else "later"
            raise HTTPException(
                status_code=409, detail=f"This meeting has not started yet; it starts at {when}"
            )
    elif meeting.status != "live":
        raise HTTPException(status_code=409, detail="This meeting has ended")
    if not (settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret):
        raise HTTPException(status_code=503, detail="LiveKit is not configured")

    if meeting.status == "scheduled":
        meeting = await store.start_meeting(meeting.id, datetime.now(UTC))
        if meeting.status != "live":
            raise HTTPException(status_code=409, detail="This meeting has ended")
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
    meeting = await team_meeting(store, user, meeting_id)
    if meeting.host_id != user.id:
        raise HTTPException(status_code=403, detail="Only the host can end the meeting")
    meeting, _ended = await end_live_meeting(store, meeting.id)
    # The post-meeting pipeline starts here when _ended; only one end ever gets True.
    return meeting


@router.post("/{meeting_id}/invitees")
async def invite(
    meeting_id: str,
    body: InviteRequest,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> Meeting:
    """Host only. Adds workspace members to the invitees."""
    meeting = await host_meeting(store, user, meeting_id)
    invitees = await team_invitees(
        store, meeting.team_id, meeting.host_id, [*meeting.invitee_ids, *body.person_ids]
    )
    return await store.set_invitees(meeting.id, invitees)


@router.delete("/{meeting_id}/invitees/{person_id}")
async def uninvite(
    meeting_id: str,
    person_id: str,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> Meeting:
    """Host only. Removing someone who is not invited changes nothing."""
    meeting = await host_meeting(store, user, meeting_id)
    if person_id not in meeting.invitee_ids:
        return meeting
    return await store.set_invitees(meeting.id, [i for i in meeting.invitee_ids if i != person_id])


@router.post("/{meeting_id}/ask")
async def ask_in_meeting(
    meeting_id: str, body: AskRequest, user: Person = Depends(current_user)
) -> Answer:
    """Typed question. Private answers return only to the asker; public ones also reach the room."""
    not_implemented()
