"""Realtime -> brain. Not exposed to browsers."""

from fastapi import APIRouter, Depends, HTTPException

from contracts import ChatMessage, InvokeRequest, InvokeResponse, SegmentsIngest

from ..store import NotFound, Store
from .deps import get_store, not_implemented, require_internal

router = APIRouter(prefix="/internal", tags=["internal"], dependencies=[Depends(require_internal)])


@router.post("/meetings/{meeting_id}/segments", status_code=204)
async def ingest_segments(
    meeting_id: str, body: SegmentsIngest, store: Store = Depends(get_store)
) -> None:
    """Final segments only; duplicates by seg_id are ignored. A bad batch saves nothing.

    Accepted after the host ends the meeting too, so the last words still reach the record.
    """
    try:
        await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
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
async def invoke(meeting_id: str, body: InvokeRequest) -> InvokeResponse:
    """A voice question, follow-up or @mention. The caller routes the answer by its visibility."""
    not_implemented()
