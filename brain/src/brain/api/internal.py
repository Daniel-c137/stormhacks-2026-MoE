"""Realtime -> brain. Not exposed to browsers."""

import math

from fastapi import APIRouter, Depends, HTTPException

from contracts import (
    AGENT_PARTICIPANT_ID,
    ChatMessage,
    InvokeRequest,
    InvokeResponse,
    SegmentsIngest,
    TranscriptSegment,
)
from contracts.meeting import MeetingStatus

from ..store import NotFound, SegmentConflict, Store
from .deps import get_store, not_implemented, require_internal

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

    This is the only writer of the transcript the report, tasks and Q&A read, so everything
    is checked here. Errors name the segment and the rule, never its text.
    """
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    if meeting.status not in ACCEPTING:
        raise HTTPException(status_code=409, detail="The report for this meeting already exists")
    if len(body.segments) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"At most {MAX_BATCH} segments per batch")
    speakers = {*meeting.participant_ids, AGENT_PARTICIPANT_ID}
    for segment in body.segments:
        if problem := segment_problem(segment, meeting_id, speakers):
            raise HTTPException(status_code=422, detail=f"Segment {segment.seg_id}: {problem}")
    try:
        await store.add_segments(meeting_id, body.segments)
    except SegmentConflict as e:
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
async def invoke(meeting_id: str, body: InvokeRequest) -> InvokeResponse:
    """A voice question, follow-up or @mention. The caller routes the answer by its visibility."""
    not_implemented()
