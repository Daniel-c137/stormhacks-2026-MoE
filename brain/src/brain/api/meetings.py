"""Meeting lifecycle (schedule, invite, join, end) and in-meeting questions (board -> brain)."""

import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from contracts import (
    Answer,
    AskRequest,
    CreateMeetingRequest,
    InviteRequest,
    JoinMeetingResponse,
    Meeting,
    MeetingPresence,
    Person,
    TranslationUpdate,
)

from ..agent.ask import MAX_RECENT_SEGMENTS, Question, ToolOrchestrator
from ..agent.pipeline import PipelineRunner, PostMeetingPipeline
from ..config import Settings
from ..livekit_rooms import Rooms
from ..livekit_tokens import participant_token
from ..store import Conflict, NotFound, Store
from ..zones import zone_of
from .deps import (
    ask_agent,
    current_user,
    get_orchestrator,
    get_pipeline,
    get_rooms,
    get_runner,
    get_settings,
    get_store,
    host_or_admin,
    team_meeting,
    user_team,
)

router = APIRouter(prefix="/meetings", tags=["meetings"])
logger = logging.getLogger(__name__)

MAX_DURATION_MIN = 24 * 60
MAX_TITLE = 200


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
    host_or_admin(meeting, user)
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
    if len(title) > MAX_TITLE:
        raise HTTPException(status_code=422, detail=f"Title is over {MAX_TITLE} characters")
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
        translate=body.translate,
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
            start = meeting.scheduled_start
            zone = await zone_of(store, team.id)
            when = start.astimezone(zone).isoformat() if start else "later"
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
        ttl=timedelta(seconds=settings.livekit_token_ttl_seconds),
    )
    return JoinMeetingResponse(meeting=meeting, livekit_url=settings.livekit_url, token=token)


@router.get("/{meeting_id}/presence")
async def presence(
    meeting_id: str,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    rooms: Rooms = Depends(get_rooms),
) -> MeetingPresence:
    """Who is in the meeting's LiveKit room now, for the lobby. participant_ids keeps everyone
    who ever joined, so it can't say who left. Only people who joined are listed, which leaves
    out the agent. 503 when LiveKit can't be reached: nobody is guessed present or absent."""
    meeting = await team_meeting(store, user, meeting_id)
    if meeting.status != "live":
        return MeetingPresence(person_ids=[])
    try:
        connected = set(await rooms.identities(meeting.id))
    except Exception:
        logger.warning("Could not list LiveKit room %s", meeting.id, exc_info=True)
        raise HTTPException(
            status_code=503, detail="Who is in the meeting can't be checked right now"
        ) from None
    return MeetingPresence(person_ids=[i for i in meeting.participant_ids if i in connected])


@router.post("/{meeting_id}/end")
async def end_meeting(
    meeting_id: str,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    pipeline: PostMeetingPipeline = Depends(get_pipeline),
    runner: PipelineRunner = Depends(get_runner),
    rooms: Rooms = Depends(get_rooms),
) -> Meeting:
    """The host or an admin. Closes the LiveKit room, so nobody keeps talking into a meeting being
    written up, and returns at once; the meeting is written up in the background."""
    meeting = await team_meeting(store, user, meeting_id)
    host_or_admin(meeting, user)
    meeting, ended = await end_live_meeting(store, meeting.id)
    if ended:  # only one end ever gets here; run() saves the first progress itself
        try:
            await rooms.close(meeting.id)
        except Exception:
            logger.warning("Could not close LiveKit room %s", meeting.id, exc_info=True)
        try:
            runner.start(meeting.id, pipeline.run)
        except Exception:
            # The meeting has ended either way; leave an error so the host can retry.
            logger.exception("could not start the write-up of meeting %s", meeting.id)
            await pipeline.not_started(meeting.id)
    return meeting


@router.put("/{meeting_id}/translation")
async def set_translation(
    meeting_id: str,
    body: TranslationUpdate,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> Meeting:
    """Host only: switch live translation of non-English speech (#106) before anyone joins.
    Turning it on is the host's approval for sending the meeting's non-English speech to the
    model as it's spoken. 409 once someone joined or the meeting ended: the worker reads it
    once, when it opens the meeting."""
    meeting = await team_meeting(store, user, meeting_id)
    if meeting.host_id != user.id:
        raise HTTPException(status_code=403, detail="Only the host can change translation")
    try:
        return await store.set_translate(meeting.id, body.translate)
    except Conflict:
        raise HTTPException(
            status_code=409, detail="Translation can't change once someone has joined"
        ) from None


@router.post("/{meeting_id}/invitees")
async def invite(
    meeting_id: str,
    body: InviteRequest,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> Meeting:
    """The host or an admin. Adds workspace members to the invitees."""
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
    """The host or an admin. Removing someone who is not invited changes nothing."""
    meeting = await host_meeting(store, user, meeting_id)
    if person_id not in meeting.invitee_ids:
        return meeting
    return await store.set_invitees(meeting.id, [i for i in meeting.invitee_ids if i != person_id])


@router.post("/{meeting_id}/ask")
async def ask_in_meeting(
    meeting_id: str,
    body: AskRequest,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    orchestrator: ToolOrchestrator = Depends(get_orchestrator),
) -> Answer:
    """Typed question, asked as the signed-in user, with the meeting's latest final segments as
    context. The answer returns only to the asker; the board posts a public one to the room.
    A private question and its answer are never stored, indexed or logged."""
    meeting = await team_meeting(store, user, meeting_id)
    recent = (await store.transcript(meeting.id))[-MAX_RECENT_SEGMENTS:]
    question = Question(
        id=str(uuid4()),
        team_id=meeting.team_id,
        text=body.question,
        asker_id=user.id,
        asker_name=user.name,
        visibility=body.visibility,
        meeting_id=meeting.id,
        history=body.history,
        recent=recent,
    )
    return await ask_agent(orchestrator, question)
