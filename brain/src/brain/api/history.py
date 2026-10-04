"""After the meeting: transcript, report, task review and history Q&A (board -> brain)."""

from fastapi import APIRouter, Depends

from contracts import (
    Answer,
    AskRequest,
    Decision,
    Meeting,
    Person,
    Report,
    ReportProgress,
    TaskDraft,
    TaskPushRequest,
    TaskPushResult,
    TranscriptSegment,
)

from ..store import Store
from .deps import current_user, get_store, member_meeting, not_implemented

router = APIRouter(tags=["history"])


@router.get("/meetings/{meeting_id}/transcript")
async def get_transcript(
    meeting: Meeting = Depends(member_meeting), store: Store = Depends(get_store)
) -> list[TranscriptSegment]:
    """Final segments in time order."""
    return await store.transcript(meeting.id)


@router.get("/meetings/{meeting_id}/report")
async def get_report(meeting_id: str, user: Person = Depends(current_user)) -> Report:
    not_implemented()


@router.get("/meetings/{meeting_id}/report/progress")
async def get_report_progress(
    meeting_id: str, user: Person = Depends(current_user)
) -> ReportProgress:
    not_implemented()


@router.patch("/meetings/{meeting_id}/tasks/{task_id}")
async def update_task(
    meeting_id: str, task_id: str, body: TaskDraft, user: Person = Depends(current_user)
) -> TaskDraft:
    not_implemented()


@router.post("/meetings/{meeting_id}/tasks/push")
async def push_tasks(
    meeting_id: str, body: TaskPushRequest, user: Person = Depends(current_user)
) -> list[TaskPushResult]:
    """The only path to external writes. Runs once a human approved these drafts and destination."""
    not_implemented()


@router.get("/decisions")
async def list_decisions(
    q: str | None = None, user: Person = Depends(current_user)
) -> list[Decision]:
    not_implemented()


@router.get("/tasks")
async def list_tasks(
    owner_id: str | None = None, user: Person = Depends(current_user)
) -> list[TaskDraft]:
    not_implemented()


@router.post("/ask")
async def ask_history(body: AskRequest, user: Person = Depends(current_user)) -> Answer:
    """Q&A across the team's previous meetings, GitHub and Jira, outside a live meeting."""
    not_implemented()
