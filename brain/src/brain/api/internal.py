"""Realtime -> brain. Not exposed to browsers."""

import asyncio
import math
from collections.abc import Callable
from weakref import WeakValueDictionary

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from contracts import (
    AGENT_PARTICIPANT_ID,
    AgendaTrackRequest,
    AgendaTrackResponse,
    ChatMessage,
    FactCheckRequest,
    FactCheckResponse,
    InvokeRequest,
    InvokeResponse,
    KeytermsResponse,
    SegmentsIngest,
    TranscriptSegment,
    WorkerMeetingResponse,
)
from contracts.meeting import MeetingStatus

from ..agent.ask import Question, ToolOrchestrator
from ..agent.factcheck import FactChecker
from ..agent.timekeeping import (
    NOW_SLACK_S,
    ClassificationFailed,
    seconds_since_start,
    track_agenda,
)
from ..config import Settings
from ..keyterms import meeting_keyterms
from ..llm import LLM, LLMError, LLMUnavailable
from ..store import Conflict, NotFound, Store
from .deps import (
    ask_agent,
    get_fact_checker,
    get_llm_factory,
    get_orchestrator,
    get_settings,
    get_store,
    require_internal,
)

router = APIRouter(prefix="/internal", tags=["internal"], dependencies=[Depends(require_internal)])


MAX_BATCH = 200
MAX_TEXT = 5000
# Live, or just ended while the worker flushes its last finals. Once the report exists the
# transcript it cites must not change underneath it.
ACCEPTING: set[MeetingStatus] = {"live", "processing"}


@router.post("/meetings/{meeting_id}/segments", status_code=204)
async def ingest_segments(
    meeting_id: str, body: SegmentsIngest, store: Store = Depends(get_store)
) -> None:
    """Final segments only; an identical resend is ignored. A bad batch saves nothing.

    This is the only writer of the transcript the report, tasks and Q&A read, so everything is
    checked here; errors name the segment and the rule, never its text. Accepted after the host
    ends the meeting too, so the last words still reach the record, but not once the report
    exists, nor once retention deleted the transcript: a late resend must not bring it back.
    """
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    if meeting.transcript_deleted_at is not None:
        raise HTTPException(status_code=409, detail="The transcript was deleted after retention")
    if meeting.status not in ACCEPTING:
        raise HTTPException(
            status_code=409, detail=f"This meeting is {meeting.status}; it takes no new segments"
        )
    if len(body.segments) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"At most {MAX_BATCH} segments per batch")
    speakers = {*meeting.participant_ids, AGENT_PARTICIPANT_ID}
    for segment in body.segments:
        if problem := segment_problem(segment, meeting_id, speakers):
            raise HTTPException(status_code=422, detail=f"Segment {segment.seg_id}: {problem}")
    try:
        await store.add_segments(meeting_id, body.segments)
    except Conflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from None


def segment_problem(segment: TranscriptSegment, meeting_id: str, speakers: set[str]) -> str | None:
    if segment.meeting_id != meeting_id:
        return "belongs to another meeting"
    if not segment.is_final:
        return "only final segments are saved"
    if not segment.text.strip():
        return "text is blank"
    if len(segment.text) > MAX_TEXT:
        return f"text is over {MAX_TEXT} characters"
    if not (math.isfinite(segment.t_start) and math.isfinite(segment.t_end)):
        return "times must be finite"
    if not 0 <= segment.t_start <= segment.t_end:
        return "times must satisfy 0 <= t_start <= t_end"
    if segment.speaker_id not in speakers:
        return "speaker is not a participant of this meeting (expected an account id)"
    return None


@router.get("/meetings/{meeting_id}")
async def worker_meeting(
    meeting_id: str, store: Store = Depends(get_store)
) -> WorkerMeetingResponse:
    """The worker reads this when it is dispatched to a room (named by the meeting id), to put
    segment times on the meeting's clock and to leave rooms that are not live meetings, and again
    before it speaks, for the team's chosen voice."""
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    settings = await store.settings(meeting.team_id)
    return WorkerMeetingResponse(meeting=meeting, voice_id=settings.voice)


@router.post("/meetings/{meeting_id}/chat", status_code=204)
async def ingest_public_chat(
    meeting_id: str, body: ChatMessage, store: Store = Depends(get_store)
) -> None:
    """Public chat only, from a participant or the agent; an identical resend (same id) is
    ignored. Private chat never reaches storage: it is refused, never saved. Taken while segments
    are, so the last messages land after the host ends the meeting."""
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    if body.visibility != "public" or body.recipient_id is not None:
        raise HTTPException(status_code=422, detail="Private chat is never stored")
    if meeting.status not in ACCEPTING or meeting.transcript_deleted_at is not None:
        raise HTTPException(
            status_code=409, detail=f"This meeting is {meeting.status}; it takes no new chat"
        )
    if body.meeting_id != meeting_id:
        raise HTTPException(status_code=422, detail="The message belongs to another meeting")
    if body.sender_id not in {*meeting.participant_ids, AGENT_PARTICIPANT_ID}:
        raise HTTPException(status_code=422, detail="The sender is not in this meeting")
    if body.is_agent != (body.sender_id == AGENT_PARTICIPANT_ID):
        raise HTTPException(status_code=422, detail="is_agent must match the sender")
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="The message is blank")
    if len(body.text) > MAX_TEXT:
        raise HTTPException(status_code=422, detail=f"The message is over {MAX_TEXT} characters")
    await store.add_public_chat(body)


@router.post("/meetings/{meeting_id}/invoke")
async def invoke(
    meeting_id: str,
    body: InvokeRequest,
    store: Store = Depends(get_store),
    orchestrator: ToolOrchestrator = Depends(get_orchestrator),
) -> InvokeResponse:
    """A voice question, follow-up or @mention, answered for the meeting's team with the recent
    final segments as context. The worker routes the answer by the invocation's visibility: a
    public one to the room, a private one only to the asker. Nothing is stored."""
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    invocation = body.invocation
    if invocation.meeting_id != meeting.id:
        raise HTTPException(status_code=422, detail="The invocation belongs to another meeting")
    if any(s.meeting_id != meeting.id for s in body.recent_segments):
        raise HTTPException(status_code=422, detail="Segment belongs to another meeting")
    question = Question(
        id=invocation.id,
        team_id=meeting.team_id,
        text=invocation.question,
        asker_id=invocation.asked_by_id,
        asker_name=invocation.asked_by_name,
        visibility=invocation.visibility,
        meeting_id=meeting.id,
        recent=body.recent_segments,
    )
    return InvokeResponse(answer=await ask_agent(orchestrator, question))


@router.get("/meetings/{meeting_id}/keyterms")
async def keyterms(
    meeting_id: str,
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> KeytermsResponse:
    """The worker fetches these when it joins a meeting and passes them to every Scribe stream
    it opens, reopens included: the agent's name first, then the team, its settings, the agenda
    and open Jira issue keys, within Scribe Realtime's limits. Jira failing never fails this."""
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    return KeytermsResponse(terms=await meeting_keyterms(store, settings, meeting))


def meeting_lock(request: Request, kind: str, meeting_id: str) -> asyncio.Lock:
    """One lock per meeting and kind of tick in this process, so overlapping ticks never handle
    a stretch twice. Kept only while someone holds or waits for it."""
    state = request.app.state
    if not hasattr(state, "tick_locks"):
        state.tick_locks = WeakValueDictionary()
    locks: WeakValueDictionary[tuple[str, str], asyncio.Lock] = state.tick_locks
    lock = locks.get((kind, meeting_id))
    if lock is None:
        lock = locks[(kind, meeting_id)] = asyncio.Lock()
    return lock


@router.post("/meetings/{meeting_id}/agenda/track")
async def track_agenda_tick(
    meeting_id: str,
    request: Request,
    body: AgendaTrackRequest | None = None,
    store: Store = Depends(get_store),
    make_llm: Callable[[], LLM] = Depends(get_llm_factory),
) -> AgendaTrackResponse:
    """The worker's timer tick (every 30-60 s, never per utterance) for a live meeting. Reads the
    final segments since the last tick from the store, so the worker sends none. The worker
    publishes the agenda on Topic.AGENDA and each nudge on Topic.AGENDA_NUDGE; nothing is spoken.

    Gemini is needed only when there is a stretch to classify: 503 when it is not configured,
    502 when the call fails. Either way nothing is tracked, and the body still carries the agenda
    and the rule nudges (saved as sent) for the worker to publish.
    """
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    if meeting.status != "live":
        raise HTTPException(status_code=409, detail="Only a live meeting keeps time")
    elapsed = seconds_since_start(meeting)
    if body and body.now is not None and body.now > elapsed + NOW_SLACK_S:
        raise HTTPException(
            status_code=422,
            detail=f"now is {body.now:.0f} s, but the meeting started {elapsed:.0f} s ago",
        )
    now = body.now if body and body.now is not None else elapsed
    async with meeting_lock(request, "agenda", meeting_id):
        try:
            return await track_agenda(store, make_llm, meeting, now)
        except ClassificationFailed as e:
            unavailable = isinstance(e.error, LLMUnavailable)
            detail = str(e.error) if unavailable else f"Could not track the agenda: {e.error}"
            return JSONResponse(
                status_code=503 if unavailable else 502,
                content={"detail": detail, **e.response.model_dump(mode="json")},
            )
        except Conflict:
            raise HTTPException(
                status_code=409, detail="The agenda kept changing; the next tick retries"
            ) from None


@router.post("/meetings/{meeting_id}/fact-check")
async def fact_check_tick(
    meeting_id: str,
    request: Request,
    body: FactCheckRequest | None = None,
    store: Store = Depends(get_store),
    checker: FactChecker = Depends(get_fact_checker),
) -> FactCheckResponse:
    """The worker's timer tick (every 30-60 s, never per utterance) for a live meeting. Reads the
    final segments since the last tick from the store, so the worker sends none. The worker
    publishes each check on Topic.FACT_CHECK, a private one only to its recipient, and
    agent_state on Topic.AGENT_STATE; nothing is spoken."""
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    if meeting.status != "live":
        raise HTTPException(status_code=409, detail="Only a live meeting is fact-checked")
    now = body.now if body and body.now is not None else seconds_since_start(meeting)
    async with meeting_lock(request, "fact-check", meeting_id):
        try:
            return await checker.tick(meeting, now)
        except LLMUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e)) from None
        except LLMError as e:
            raise HTTPException(status_code=502, detail=f"Could not fact-check: {e}") from e
