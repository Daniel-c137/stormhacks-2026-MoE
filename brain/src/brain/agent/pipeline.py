"""The write-up after a meeting ends: settle the transcript, build the report, link past
decisions, save (processing -> needs_review), then index memory. Private chat is never an input.

What a failure leaves behind:
- Before the save (settling, writing, linking, or the save itself): ReportProgress with the error
  at the failed step, the meeting still processing, and nothing else. The save is one atomic
  Store.complete_report: the report, its decision links and the status change land together or
  not at all, and it first undoes any links an earlier save of this meeting made, so a retry
  never leaves a past decision retired by a decision that no longer exists.
- Indexing runs after the save. A failure there leaves the meeting in review with its report
  and no memory for it (a meeting's memory is swapped atomically), and the step's label says
  indexing failed. Memory is never written for a meeting whose report was not saved.

The team sees a message chosen for them; the exception itself goes to the server log."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Protocol

from contracts import Decision, Report, ReportProgress

from ..llm import LLM, LLMError, LLMUnavailable
from ..memory import MeetingMemory, MemoryMisconfigured, UnusableMemory
from ..report import TranscriptInput, build_report, link_decisions, memory_candidates
from ..store import Store
from ..zones import local_date, zone_of

logger = logging.getLogger(__name__)

REPORT_STEPS = [
    "Finalising the transcript",
    "Writing the summary, decisions and tasks",
    "Checking against past decisions",
    "Saving the report",
    "Indexing for search",
]
SETTLE, WRITE, LINK, SAVE, INDEX = range(len(REPORT_STEPS))
INDEX_SKIPPED = "Indexing for search (skipped: search memory is not configured)"
NO_TRANSCRIPT = "No transcript was captured for this meeting, so there is nothing to summarise."
INTERRUPTED = "The write-up was interrupted before it finished; retry it"
NOT_STARTED = "The write-up could not be started; retry it"

Memory = MeetingMemory | UnusableMemory


class PostMeetingPipeline(Protocol):
    """Runs once a meeting ends: summary, decisions, task drafts, past-decision links, indexing.

    Private chat is never an input. The first step waits for the worker to flush its last
    final segments (or a short grace period) before reading the transcript: segments are
    still accepted while the meeting is `processing`, and refused once the report exists.
    """

    steps: list[str]

    def progress(self, meeting_id: str) -> ReportProgress:
        """The first step, under way. Not saved: run() saves it."""
        ...

    async def not_started(self, meeting_id: str) -> None:
        """Records that the write-up could not be started, so it can be retried."""
        ...

    async def run(self, meeting_id: str) -> Report: ...


class WriteUpError(Exception):
    """A step cannot go on; the message is shown to the team."""


def can_retry(progress: ReportProgress | None, now: datetime, stale_after: timedelta) -> bool:
    """For a processing meeting whose write-up is not running in this process: retry when the
    last run failed, never saved progress, or has not moved for `stale_after` (its process
    probably died)."""
    if progress is None or progress.error is not None or progress.updated_at is None:
        return True
    return now - progress.updated_at >= stale_after


def shown_error(e: BaseException, step: str) -> str:
    """What the team sees when `step` failed. Only messages written for them pass through."""
    if isinstance(e, WriteUpError | LLMUnavailable | MemoryMisconfigured):
        return str(e)
    if isinstance(e, LLMError):
        return f'The language model failed during "{step}"; retry the write-up later'
    return f'"{step}" failed unexpectedly; retry the write-up, or see the server log if it repeats'


def index_problem(e: BaseException) -> str:
    if isinstance(e, MemoryMisconfigured):
        return str(e)
    if isinstance(e, LLMError):
        return "the embedding model failed"
    return "an unexpected error; see the server log"


def reopened(decision: Decision, retired_by: set[str]) -> Decision:
    """The past decision as it was before one of `retired_by` superseded it."""
    relation = decision.relation
    if relation and relation.type == "superseded_by" and relation.decision_id in retired_by:
        return decision.model_copy(update={"status": "active", "relation": None})
    return decision


class ReportPipeline:
    """`llm` is called when the report step starts, so an unconfigured model fails that step
    with a visible error instead of the end request. Without `memory` the index step is skipped
    and says so. `settle` replaces the wait for the last transcript segments (tests gate it)."""

    def __init__(
        self,
        store: Store,
        llm: Callable[[], LLM],
        memory: Memory | None = None,
        *,
        settle_seconds: float = 0.0,
        settle: Callable[[], Awaitable[object]] | None = None,
    ):
        self.store = store
        self.llm = llm
        self.memory = memory
        self.settle_seconds = settle_seconds
        self._settle = settle or (lambda: asyncio.sleep(settle_seconds))
        self.steps = [
            INDEX_SKIPPED if i == INDEX and memory is None else step
            for i, step in enumerate(REPORT_STEPS)
        ]

    def progress(self, meeting_id: str) -> ReportProgress:
        return self._progress(meeting_id, SETTLE)

    async def not_started(self, meeting_id: str) -> None:
        await self._failed(self.progress(meeting_id), NOT_STARTED)

    async def run(self, meeting_id: str) -> Report:
        progress = self.progress(meeting_id)
        try:
            await self.store.save_report_progress(progress)
            await self._settle()
            meeting, transcript, agenda = await self._read(meeting_id)
            zone = await zone_of(self.store, meeting.team_id)
            superseded: list[Decision] = []
            # The public fact-checks from the live meeting; private ones were never stored.
            fact_checks = await self.store.fact_checks(meeting_id)
            if transcript.final_segments():
                progress = await self._at(progress, WRITE)
                llm = self.llm()
                report = await build_report(
                    llm, transcript, agenda=agenda, zone=zone, fact_checks=fact_checks
                )
                progress = await self._at(progress, LINK)
                report, superseded = await self._links(llm, meeting.team_id, report, zone)
            else:
                report = Report(meeting_id=meeting_id, summary=NO_TRANSCRIPT)
            progress = await self._at(progress, SAVE)
            report.fact_checks = fact_checks
            await self.store.complete_report(report, superseded)
        except asyncio.CancelledError:
            await self._failed(progress, INTERRUPTED)
            raise
        except Exception as e:
            await self._failed(progress, shown_error(e, self.steps[progress.current]))
            raise
        steps = await self._index(progress, meeting.team_id, transcript, report)
        try:
            await self.store.save_report_progress(
                self._progress(meeting_id, len(steps), steps=steps, done=True)
            )
        except Exception:
            logger.exception("could not record the finished write-up of %s", meeting_id)
        return report

    async def _read(self, meeting_id: str):
        """The final transcript once the worker's last segments have had time to land."""
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
        agenda = await self.store.agenda(meeting_id)
        return meeting, transcript, agenda.items if agenda else []

    async def _links(
        self, llm: LLM, team_id: str, report: Report, zone: tzinfo
    ) -> tuple[Report, list[Decision]]:
        """The report with its decisions' links, and the past decisions they retire. Past
        decisions are the team's from other meetings, as they were before any earlier save of
        this meeting retired them, dated by their meetings' days in the team's `zone`."""
        stored = await self.store.decisions(team_id)
        earlier = {d.id for d in stored if d.meeting_id == report.meeting_id}
        past = [reopened(d, earlier) for d in stored if d.meeting_id != report.meeting_id]
        if not report.decisions or not past:
            return report, []
        dates = {
            m.id: day
            for m in await self.store.meetings(team_id)
            if (day := local_date(m.started_at or m.scheduled_start, zone))
        }
        candidates = None
        if self.memory is not None:
            candidates = await memory_candidates(self.memory, team_id, report.decisions, past)
        links = await link_decisions(
            llm, report.decisions, past, dates=dates, candidates=candidates
        )
        linked = {d.id: d for d in links.new}
        decisions = [linked.get(d.id, d) for d in report.decisions]
        return report.model_copy(update={"decisions": decisions}), links.past

    async def _index(
        self, progress: ReportProgress, team_id: str, transcript: TranscriptInput, report: Report
    ) -> list[str]:
        """Runs once the report is saved. A failure is a note on the step, not a failed write-up."""
        steps = list(self.steps)
        if self.memory is None or not transcript.final_segments():
            return steps
        try:
            await self._at(progress, INDEX)
            await self.memory.index_meeting(team_id, report.meeting_id, transcript.segments, report)
        except Exception as e:
            logger.exception("indexing meeting %s failed", report.meeting_id)
            steps[INDEX] = f"{REPORT_STEPS[INDEX]} (failed: {index_problem(e)})"
        return steps

    def _progress(
        self, meeting_id: str, current: int, *, steps: list[str] | None = None, done=False
    ) -> ReportProgress:
        return ReportProgress(
            meeting_id=meeting_id,
            steps=list(steps or self.steps),
            current=current,
            done=done,
            updated_at=datetime.now(UTC),
        )

    async def _at(self, progress: ReportProgress, step: int) -> ReportProgress:
        return await self.store.save_report_progress(self._progress(progress.meeting_id, step))

    async def _failed(self, progress: ReportProgress, error: str) -> None:
        failed = progress.model_copy(update={"error": error, "updated_at": datetime.now(UTC)})
        try:
            await self.store.save_report_progress(failed)
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
        except Exception:
            logger.exception("the write-up of meeting %s failed", meeting_id)

    def _forget(self, meeting_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(meeting_id) is task:
            del self._tasks[meeting_id]
