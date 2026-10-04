"""Realtime -> brain. Not exposed to browsers."""

import asyncio
from collections.abc import Callable
from weakref import WeakValueDictionary

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from contracts import (
    AgendaTrackRequest,
    AgendaTrackResponse,
    ChatMessage,
    FactCheckRequest,
    FactCheckResponse,
    InvokeRequest,
    InvokeResponse,
    SegmentsIngest,
)

from ..agent.ask import Question, ToolOrchestrator
from ..agent.factcheck import FactChecker
from ..agent.timekeeping import (
    NOW_SLACK_S,
    ClassificationFailed,
    seconds_since_start,
    track_agenda,
)
from ..llm import LLM, LLMError, LLMUnavailable
from ..store import Conflict, NotFound, Store
from .deps import (
    ask_agent,
    get_fact_checker,
    get_llm_factory,
    get_orchestrator,
    get_store,
    not_implemented,
    require_internal,
)

router = APIRouter(prefix="/internal", tags=["internal"], dependencies=[Depends(require_internal)])


@router.post("/meetings/{meeting_id}/segments", status_code=204)
async def ingest_segments(
    meeting_id: str, body: SegmentsIngest, store: Store = Depends(get_store)
) -> None:
    """Final segments only; duplicates by seg_id are ignored. A bad batch saves nothing.

    Accepted after the host ends the meeting too, so the last words still reach the record, but
    not once retention deleted the transcript: a late resend must not bring it back.
    """
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    if meeting.transcript_deleted_at is not None:
        raise HTTPException(status_code=409, detail="The transcript was deleted after retention")
    if any(s.meeting_id != meeting_id for s in body.segments):
        raise HTTPException(status_code=422, detail="Segment belongs to another meeting")
    if any(not s.is_final for s in body.segments):
        raise HTTPException(status_code=422, detail="Only final segments are saved")
    await store.add_segments(meeting_id, body.segments)


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
