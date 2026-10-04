"""The write-up after a meeting ends: settle the transcript, build the report, link past
decisions, index memory, save, then processing -> needs_review. Private chat is never an input.

Every model and embedding call comes before the first store write, so a failed step leaves the
meeting processing with ReportProgress.error and nothing saved; a retry starts over and replaces
the report and the meeting's memory."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from contracts import Report, ReportProgress

from ..llm import LLM
from ..memory import MeetingMemory
from ..report import (
    DecisionLinks,
    TranscriptInput,
    apply_links,
    build_report,
    link_decisions,
    memory_candidates,
)
from ..store import Store

logger = logging.getLogger(__name__)

REPORT_STEPS = [
    "Finalising the transcript",
    "Writing the summary, decisions and tasks",
    "Checking against past decisions",
    "Indexing for search",
    "Saving the report",
]
SETTLE, WRITE, LINK, INDEX, SAVE = range(len(REPORT_STEPS))
INDEX_SKIPPED = "Indexing for search (skipped: search memory is not configured)"
INTERRUPTED = "The write-up was interrupted before it finished"


class PostMeetingPipeline(Protocol):
    """Runs once a meeting ends: summary, decisions, task drafts, past-decision links, indexing.

    Private chat is never an input.
    """

    steps: list[str]

    async def queued(self, meeting_id: str) -> ReportProgress:
        """Saves the first step as under way, so readers see it before the run starts."""
        ...

    async def run(self, meeting_id: str) -> Report: ...


class WriteUpError(Exception):
    """A step cannot go on; the message is shown to the team."""


class ReportPipeline:
    """`llm` is called when the report step starts, so an unconfigured model fails that step
    with a visible error instead of the end request. Without `memory` the index step is skipped
    and says so."""

    def __init__(
        self,
        store: Store,
        llm: Callable[[], LLM],
        memory: MeetingMemory | None = None,
        *,
        settle_seconds: float = 0.0,
    ):
        self.store = store
        self.llm = llm
        self.memory = memory
        self.settle_seconds = settle_seconds
        self.steps = [
            INDEX_SKIPPED if i == INDEX and memory is None else step
            for i, step in enumerate(REPORT_STEPS)
        ]

    async def queued(self, meeting_id: str) -> ReportProgress:
        return await self.store.save_report_progress(self._progress(meeting_id, SETTLE))

    async def run(self, meeting_id: str) -> Report:
        progress = await self.queued(meeting_id)
        try:
            meeting, transcript, agenda = await self._settled(meeting_id)
            progress = await self._at(meeting_id, WRITE)
            llm = self.llm()
            report = await build_report(llm, transcript, agenda=agenda)
            progress = await self._at(meeting_id, LINK)
            links = await self._links(llm, meeting.team_id, report)
            progress = await self._at(meeting_id, INDEX)
            if self.memory is not None:
                await self.memory.index_meeting(
                    meeting.team_id, meeting_id, transcript.segments, report
                )
            progress = await self._at(meeting_id, SAVE)
            # The public fact-checks from the live meeting; private ones were never stored.
            report.fact_checks = await self.store.fact_checks(meeting_id)
            await self.store.save_report(report)
            await apply_links(self.store, links)
            await self.store.transition_status(meeting_id, {"processing"}, "needs_review")
        except asyncio.CancelledError:
            await self._failed(progress, INTERRUPTED)
            raise
        except Exception as e:
            await self._failed(progress, str(e) or type(e).__name__)
            raise
        await self.store.save_report_progress(
            self._progress(meeting_id, len(self.steps), done=True)
        )
        return report

    async def _settled(self, meeting_id: str):
        """The final transcript once the worker's last segments have had time to land."""
        if self.settle_seconds > 0:
            await asyncio.sleep(self.settle_seconds)
        meeting = await self.store.meeting(meeting_id)
        if meeting.status != "processing":
            raise WriteUpError(f"The meeting is {meeting.status}, not processing")
        transcript = TranscriptInput(
            meeting_id=meeting.id,
            title=meeting.title,
            started_at=meeting.started_at,
            members=await self.store.members(meeting.team_id),
            segments=await self.store.transcript(meeting_id),
        )
        if not transcript.final_segments():
            raise WriteUpError("No final transcript was saved for this meeting")
        agenda = await self.store.agenda(meeting_id)
        return meeting, transcript, agenda.items if agenda else []

    async def _links(self, llm: LLM, team_id: str, report: Report) -> DecisionLinks:
        """Links to the team's decisions from other meetings, dated by those meetings."""
        if not report.decisions:
            return DecisionLinks()
        past = [d for d in await self.store.decisions(team_id) if d.meeting_id != report.meeting_id]
        if not past:
            return DecisionLinks()
        dates = {
            m.id: when.date()
            for m in await self.store.meetings(team_id)
            if (when := m.started_at or m.scheduled_start)
        }
        candidates = None
        if self.memory is not None:
            candidates = await memory_candidates(self.memory, team_id, report.decisions, past)
        return await link_decisions(llm, report.decisions, past, dates=dates, candidates=candidates)

    def _progress(self, meeting_id: str, current: int, *, done: bool = False) -> ReportProgress:
        return ReportProgress(
            meeting_id=meeting_id, steps=list(self.steps), current=current, done=done
        )

    async def _at(self, meeting_id: str, step: int) -> ReportProgress:
        return await self.store.save_report_progress(self._progress(meeting_id, step))

    async def _failed(self, progress: ReportProgress, error: str) -> None:
        try:
            await self.store.save_report_progress(progress.model_copy(update={"error": error}))
        except Exception:
            logger.exception("could not record the failed write-up of %s", progress.meeting_id)


class PipelineRunner:
    """Write-ups in the background, at most one per meeting at a time. One per app, on
    app.state.pipeline_runner. Tests await drain()."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def running(self, meeting_id: str) -> bool:
        task = self._tasks.get(meeting_id)
        return task is not None and not task.done()

    def start(self, meeting_id: str, run: Callable[[str], Awaitable[object]]) -> bool:
        """False, and nothing started, when the meeting's write-up is already running."""
        if self.running(meeting_id):
            return False
        task = asyncio.create_task(self._run(meeting_id, run), name=f"write-up {meeting_id}")
        self._tasks[meeting_id] = task
        task.add_done_callback(lambda done: self._forget(meeting_id, done))
        return True

    async def drain(self) -> None:
        """Waits until no write-up is running, including any started meanwhile."""
        while pending := [t for t in self._tasks.values() if not t.done()]:
            await asyncio.wait(pending)

    async def shutdown(self) -> None:
        """Stops every running write-up; each records that it was interrupted, so it can be
        retried."""
        for task in self._tasks.values():
            task.cancel()
        await self.drain()

    async def _run(self, meeting_id: str, run: Callable[[str], Awaitable[object]]) -> None:
        try:
            await run(meeting_id)
        except Exception as e:
            logger.warning("the write-up of meeting %s failed: %s", meeting_id, e)

    def _forget(self, meeting_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(meeting_id) is task:
            del self._tasks[meeting_id]
