"""After the meeting: transcript, report, task review and history Q&A (board -> brain)."""

from collections.abc import Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from contracts import (
    Answer,
    AskRequest,
    Decision,
    Person,
    Report,
    ReportProgress,
    TaskDraft,
    TaskPushRequest,
    TaskPushResult,
    TranscriptSegment,
)

from ..agent.ask import Question, ToolOrchestrator
from ..jira import ApprovalRequired, JiraPusher, JiraUnavailable, apply_results
from ..report import ProcessedMeeting
from ..store import Conflict, NotFound, Store
from .deps import (
    ask_agent,
    current_user,
    get_jira_pusher,
    get_orchestrator,
    get_store,
    team_meeting,
    user_team,
)

router = APIRouter(tags=["history"])

REVIEWABLE = {"needs_review", "pushed"}


@router.get("/meetings/{meeting_id}/transcript")
async def get_transcript(
    meeting_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> list[TranscriptSegment]:
    """Final segments in time order."""
    meeting = await team_meeting(store, user, meeting_id)
    return await store.transcript(meeting.id)


@router.get("/meetings/{meeting_id}/report")
async def get_report(
    meeting_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Report:
    """With the tasks and decisions as edited so far. 404 until the pipeline saved it."""
    meeting = await team_meeting(store, user, meeting_id)
    try:
        return await store.report(meeting.id)
    except NotFound:
        raise HTTPException(status_code=404, detail="No report yet") from None


@router.get("/meetings/{meeting_id}/report/progress")
async def get_report_progress(
    meeting_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> ReportProgress:
    meeting = await team_meeting(store, user, meeting_id)
    progress = await store.report_progress(meeting.id)
    if progress is None:
        raise HTTPException(status_code=404, detail="The report has not been started")
    return progress


@router.patch("/meetings/{meeting_id}/tasks/{task_id}")
async def update_task(
    meeting_id: str,
    task_id: str,
    body: TaskDraft,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> TaskDraft:
    """A reviewer's edit. Only the title, description, owner, due date and include flag change;
    where the task came from and its Jira state are the pipeline's and Jira's."""
    meeting = await team_meeting(store, user, meeting_id)
    try:
        task = await store.task(meeting.team_id, task_id)
    except NotFound:
        task = None
    if task is None or task.meeting_id != meeting.id:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.key:
        raise HTTPException(status_code=409, detail=f"Already pushed as {task.key}")
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title is required")
    if body.owner_id is not None:
        team = await store.team(meeting.team_id)
        if body.owner_id not in team.member_ids:
            raise HTTPException(status_code=422, detail="The owner must be on the team")
    edited = task.model_copy(
        update={
            "title": title,
            "description": body.description,
            "owner_id": body.owner_id,
            "due": body.due,
            "include": body.include,
        }
    )
    return await store.update_task(edited)


@router.post("/meetings/{meeting_id}/tasks/push")
async def push_tasks(
    meeting_id: str,
    body: TaskPushRequest,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    make_pusher: Callable[[], JiraPusher] = Depends(get_jira_pusher),
) -> list[TaskPushResult]:
    """The only path to external writes. Runs once a human approved these drafts and destination.
    The approver is always the caller. The meeting leaves review once every included draft has
    a key; a draft Jira rejected keeps it in review so it can be fixed and pushed again."""
    meeting = await team_meeting(store, user, meeting_id)
    if meeting.status not in REVIEWABLE:
        raise HTTPException(
            status_code=409, detail=f"The meeting is {meeting.status}, not in review"
        )
    try:
        pusher = make_pusher()
    except JiraUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    try:
        report = await store.report(meeting.id)
    except NotFound:
        raise HTTPException(status_code=404, detail="No report yet") from None

    review = ProcessedMeeting(
        meeting_id=meeting.id,
        title=meeting.title,
        started_at=meeting.started_at,
        members=await store.members(meeting.team_id),
        report=report,
    )
    request = body.model_copy(update={"approved_by": user.name})
    try:
        results = await pusher.push(review, request)
    except ApprovalRequired as e:
        raise HTTPException(status_code=422, detail=str(e)) from None

    tasks = apply_results(report.tasks, results)
    for before, after in zip(report.tasks, tasks, strict=True):
        if after != before:
            await store.update_task(after)
    new_keys = [r.key for r in results if r.key and r.key not in meeting.jira_keys]
    if new_keys:
        keys = list(dict.fromkeys([*meeting.jira_keys, *new_keys]))
        meeting = await store.update_meeting(meeting.model_copy(update={"jira_keys": keys}))
    if all(task.key for task in tasks if task.include):
        try:
            await store.transition_status(meeting.id, {"needs_review"}, "pushed")
        except Conflict:
            pass  # already pushed
    return results


@router.get("/decisions")
async def list_decisions(
    q: str | None = None, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> list[Decision]:
    """The team's decisions, newest first, with their superseded/contradicts relations."""
    team = await user_team(store, user)
    return await store.decisions(team.id, q)


@router.get("/tasks")
async def list_tasks(
    owner_id: str | None = None,
    open: bool = False,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> list[TaskDraft]:
    """Drafts and pushed tasks with their Jira status; `open` leaves out the done ones."""
    team = await user_team(store, user)
    tasks = await store.tasks(team.id, owner_id)
    return [t for t in tasks if t.jira_status != "done"] if open else tasks


@router.post("/ask")
async def ask_history(
    body: AskRequest,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    orchestrator: ToolOrchestrator = Depends(get_orchestrator),
) -> Answer:
    """Home's chat: Q&A across the caller's team's meetings, decisions, tasks, GitHub and Jira,
    with `history` for follow-ups. Nothing about the question is stored."""
    team = await user_team(store, user)
    question = Question(
        id=str(uuid4()),
        team_id=team.id,
        text=body.question,
        asker_id=user.id,
        asker_name=user.name,
        visibility=body.visibility,
        history=body.history,
    )
    return await ask_agent(orchestrator, question)
