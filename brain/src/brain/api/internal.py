"""Realtime -> brain. Not exposed to browsers."""

from fastapi import APIRouter, Depends

from contracts import ChatMessage, InvokeRequest, InvokeResponse, SegmentsIngest

from .deps import not_implemented, require_internal

router = APIRouter(prefix="/internal", tags=["internal"], dependencies=[Depends(require_internal)])


@router.post("/meetings/{meeting_id}/segments", status_code=204)
async def ingest_segments(meeting_id: str, body: SegmentsIngest) -> None:
    """Final segments only; duplicates by seg_id are ignored."""
    not_implemented()


@router.post("/meetings/{meeting_id}/chat", status_code=204)
async def ingest_public_chat(meeting_id: str, body: ChatMessage) -> None:
    """Public chat only. Private chat never reaches storage."""
    not_implemented()


@router.post("/meetings/{meeting_id}/invoke")
async def invoke(meeting_id: str, body: InvokeRequest) -> InvokeResponse:
    """A voice question, follow-up or @mention. The caller routes the answer by its visibility."""
    not_implemented()
