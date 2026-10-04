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
    InvokeRequest,
    InvokeResponse,
    SegmentsIngest,
    TranscriptSegment,
)
from contracts.meeting import MeetingStatus

from ..agent.ask import Question, ToolOrchestrator
from ..agent.timekeeping import (
    NOW_SLACK_S,
    ClassificationFailed,
    seconds_since_start,
    track_agenda,
)
from ..llm import LLM, LLMUnavailable
from ..store import Conflict, NotFound, Store
from .deps import (
    ask_agent,
    get_llm_factory,
    get_orchestrator,
    get_store,
    not_implemented,
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


@router.post("/meetings/{meeting_id}/chat", status_code=204)
async def ingest_public_chat(meeting_id: str, body: ChatMessage) -> None:
    """Public chat only. Private chat never reaches storage."""
    not_implemented()


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


def agenda_lock(request: Request, meeting_id: str) -> asyncio.Lock:
    """One lock per meeting in this process, so overlapping ticks never classify a stretch twice.
    Kept only while someone holds or waits for it."""
    state = request.app.state
    if not hasattr(state, "agenda_locks"):
        state.agenda_locks = WeakValueDictionary()
    locks: WeakValueDictionary[str, asyncio.Lock] = state.agenda_locks
    lock = locks.get(meeting_id)
    if lock is None:
        lock = locks[meeting_id] = asyncio.Lock()
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
    async with agenda_lock(request, meeting_id):
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
