"""After the meeting: transcript, report, task review and history Q&A (board -> brain)."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response

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
from ..agent.pipeline import PipelineRunner, PostMeetingPipeline, can_retry
from ..config import Settings
from ..jira import ApprovalRequired, JiraPusher, JiraUnavailable, apply_results
from ..report import ProcessedMeeting
from ..speech import (
    CONTENT_TYPE,
    MeetingLocks,
    SpeechFailed,
    SpeechUnavailable,
    audio_key,
    missing,
    synthesize,
)
from ..store import Conflict, NotFound, Store
from .deps import (
    ask_agent,
    current_user,
    get_http_transport,
    get_jira_pusher,
    get_orchestrator,
    get_pipeline,
    get_runner,
    get_settings,
    get_speech_locks,
    get_store,
    team_meeting,
    user_team,
)

router = APIRouter(tags=["history"])

REVIEWABLE = {"needs_review", "pushed"}
RUNNING = "The write-up is still running"


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


@router.get("/meetings/{meeting_id}/report/audio")
async def get_report_audio(
    meeting_id: str,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_settings),
    transport: httpx.AsyncBaseTransport | None = Depends(get_http_transport),
    locks: MeetingLocks = Depends(get_speech_locks),
) -> Response:
    """The report's summary read aloud, word for word, in the team's agent voice (else
    ELEVENLABS_VOICE_ID), as MP3. Made with ElevenLabs once, then served from the store until the
    summary, voice or model changes. For the report page only; never played into a meeting."""
    meeting = await team_meeting(store, user, meeting_id)
    if meeting.status == "processing":
        raise HTTPException(status_code=409, detail=RUNNING)
    try:
        report = await store.report(meeting.id)
    except NotFound:
        raise HTTPException(status_code=404, detail="No report yet") from None
    text = report.summary
    if not text.strip():
        raise HTTPException(status_code=404, detail="The summary is empty: nothing to read")
    voice = (await store.settings(meeting.team_id)).voice or settings.elevenlabs_voice_id
    model = settings.elevenlabs_tts_model
    gaps = missing(settings, voice)
    if gaps or not voice or not model:
        raise HTTPException(
            status_code=503, detail=f"ElevenLabs speech is not configured: set {', '.join(gaps)}"
        )
    key = audio_key(text, voice, model)
    async with locks.hold(meeting.id):
        try:
            audio = await store.report_audio(meeting.id)
        except NotFound:
            audio = None
        if audio is None or audio.key != key:
            try:
                data = await synthesize(settings, voice, text, transport=transport)
            except SpeechUnavailable as e:
                raise HTTPException(status_code=503, detail=str(e)) from None
            except SpeechFailed as e:
                raise HTTPException(status_code=502, detail=str(e)) from None
            audio = await store.save_report_audio(meeting.id, key, CONTENT_TYPE, data)
    return Response(
        content=audio.data,
        media_type=audio.content_type,
        headers={"Cache-Control": "private, no-cache", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/meetings/{meeting_id}/report/progress")
async def get_report_progress(
    meeting_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> ReportProgress:
    meeting = await team_meeting(store, user, meeting_id)
    progress = await store.report_progress(meeting.id)
    if progress is None:
        raise HTTPException(status_code=404, detail="The report has not been started")
    return progress


@router.post("/meetings/{meeting_id}/report/retry", status_code=202)
async def retry_report(
    meeting_id: str,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    pipeline: PostMeetingPipeline = Depends(get_pipeline),
    runner: PipelineRunner = Depends(get_runner),
    settings: Settings = Depends(get_settings),
) -> ReportProgress:
    """Host only, for a meeting still being written up that nothing here is running: once the
    last run failed, never saved progress, or has made no progress for PIPELINE_STALE_MINUTES
    (its process died). Starts it over; the report, its links and the meeting's memory are
    replaced, not added to. A refused retry changes nothing."""
    meeting = await team_meeting(store, user, meeting_id)
    if meeting.host_id != user.id:
        raise HTTPException(status_code=403, detail="Only the host can retry the write-up")
    if meeting.status != "processing":
        raise HTTPException(
            status_code=409, detail=f"The meeting is {meeting.status}, not being written up"
        )
    if runner.running(meeting.id):
        raise HTTPException(status_code=409, detail=RUNNING)
    progress = await store.report_progress(meeting.id)
    minutes = settings.pipeline_stale_minutes
    if not can_retry(progress, datetime.now(UTC), timedelta(minutes=minutes)):
        raise HTTPException(
            status_code=409,
            detail="The write-up is still in progress; it can be retried once it fails or makes"
            f" no progress for {minutes:g} minutes",
        )
    if not runner.start(meeting.id, pipeline.run):
        raise HTTPException(status_code=409, detail=RUNNING)
    return pipeline.progress(meeting.id)


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
