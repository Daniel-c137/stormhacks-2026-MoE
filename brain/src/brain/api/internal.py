"""Realtime -> brain. Not exposed to browsers."""

from fastapi import APIRouter, Depends, HTTPException

from contracts import ChatMessage, InvokeRequest, InvokeResponse, SegmentsIngest

from ..agent.ask import Question, ToolOrchestrator
from ..store import NotFound, Store
from .deps import ask_agent, get_orchestrator, get_store, not_implemented, require_internal

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
