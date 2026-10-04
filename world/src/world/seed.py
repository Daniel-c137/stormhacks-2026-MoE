"""`world-seed`: the demo world's team, people, settings and past meetings, loaded into a migrated
database through the brain's own store and written up by the brain's own pipeline.

Each past meeting goes through the lifecycle a real one does: scheduled, started at its real
time with the team in it, its transcript saved, ended at its real time, written up (report,
decisions linked to the earlier meetings', tasks, meeting memory), and then reviewed. The review
pushes nothing: no Jira issue or GitHub write is ever made. Meetings go in date order, so a later
meeting's decisions can supersede an earlier one's.

Running it again changes nothing: a meeting already seeded is skipped, and one a failed run left
part-way is finished. `--reset` first removes the seeded team and everything under it."""

import argparse
import asyncio
import re
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from typing import Literal

from pydantic import BaseModel

from brain.agent.pipeline import INDEX, REPORT_STEPS, ReportPipeline
from brain.config import Settings as BrainSettings
from brain.db import migrate, open_pool
from brain.llm import (
    LLM,
    Embedder,
    Embeddings,
    EmbedTask,
    LLMError,
    LLMUnavailable,
    OpenRouterLLM,
    make_embedder,
    make_llm,
)
from brain.memory import MeetingMemory, PgMemoryStore
from brain.memory.pg import DIM
from brain.pg_store import PostgresStore
from brain.report.models import person_from_name
from brain.store import NotFound, Store
from contracts import (
    GitHubSettings,
    JiraSettings,
    Meeting,
    Person,
    ReportProgress,
    Team,
    TeamSettings,
)

from .config import Settings, SnapshotName, world_spec
from .past_meetings import SeedMeeting, load_meetings, person_id, snapshot_meetings

EMAIL_DOMAIN = "dropsubs.example"
"""Non-routable (RFC 2606), and the domain the world's mock Jira gives its people."""
JIRA_SITE = "https://dropsubs.atlassian.net"
"""The world's Jira site, as mock-data/jira records it; JIRA_BASE_URL wins when set."""
TEAM_TIMEZONE = "America/Vancouver"

Outcome = Literal["seeded", "resumed", "skipped"]


class SeedError(RuntimeError):
    """A meeting could not be seeded; what was saved stays, and the next run finishes it."""


class Seeded(BaseModel):
    """One past meeting's result. `models` answered its write-up; `cost` is what OpenRouter
    reported for it in USD, None when OpenRouter did not answer."""

    meeting_id: str
    day: date
    title: str
    outcome: Outcome
    models: list[str] = []
    cost: float | None = None


def team_id(company: str) -> str:
    return f"team-{re.sub(r'[^a-z0-9]+', '-', company.casefold()).strip('-')}"


def seed_people(meetings: Sequence[SeedMeeting] | None = None) -> list[Person]:
    """The world's people, from the meetings' participants (by default every meeting file), in
    order of first appearance. Logins are added separately."""
    people: dict[str, Person] = {}
    for meeting in load_meetings() if meetings is None else meetings:
        for participant in meeting.participants:
            pid = person_id(participant.name)
            if pid not in people:
                local = re.sub(r"[^a-z0-9]", "", participant.name.casefold())
                people[pid] = person_from_name(pid, participant.name).model_copy(
                    update={"title": participant.role, "email": f"{local}@{EMAIL_DOMAIN}"}
                )
    return list(people.values())


async def seed_team(store: Store, people: Sequence[Person], *, jira_site: str = JIRA_SITE) -> Team:
    """The world's team and its people. Its settings are written only when the team is new, so
    a later change in the app survives another seed."""
    spec = world_spec()
    tid = team_id(spec.company)
    try:
        await store.team(tid)
        new = False
    except NotFound:
        await store.create_team(
            Team(
                id=tid,
                name=spec.company,
                member_ids=[],
                github_repo=spec.github_repo,
                jira_project=spec.jira_project,
            )
        )
        new = True
    for person in people:
        await store.upsert_person(person, tid)
    if new:
        await store.save_settings(
            TeamSettings(
                team_id=tid,
                github=GitHubSettings(repo=spec.github_repo),
                jira=JiraSettings(site=jira_site, project=spec.jira_project),
                timezone=TEAM_TIMEZONE,
            )
        )
    return await store.team(tid)


async def reset_team(store: Store, memory: MeetingMemory) -> int:
    """Removes the seeded team, everything under it and its meetings' memory. Returns how many
    meetings went; 0 when there was no team."""
    tid = team_id(world_spec().company)
    try:
        meetings = await store.meetings(tid)
    except NotFound:
        return 0
    for meeting in meetings:  # memory is not the store's, so it goes first, meeting by meeting
        await memory.store.replace_meeting(meeting.id, [], Embeddings(model="none", vectors=[]))
    try:
        await store.delete_team(tid)
    except NotFound:
        return 0
    return len(meetings)


async def seed_world(
    store: Store,
    memory: MeetingMemory,
    make_llm: Callable[[], LLM],
    meetings: Sequence[SeedMeeting],
    *,
    reset: bool = False,
    jira_site: str = JIRA_SITE,
    on_seeded: Callable[[Seeded], object] | None = None,
) -> list[Seeded]:
    """Seeds the team and `meetings`, oldest first. Raises SeedError at the first meeting that
    fails, after reporting the ones before it."""
    if reset:
        await reset_team(store, memory)
    team = await seed_team(store, seed_people(load_meetings()), jira_site=jira_site)
    results: list[Seeded] = []
    for meeting in sorted(meetings, key=lambda m: m.started_at):
        result = await seed_meeting(store, memory, make_llm, team, meeting)
        results.append(result)
        if on_seeded is not None:
            on_seeded(result)
    return results


async def seed_meetings(
    name: SnapshotName,
    store: Store,
    memory: MeetingMemory,
    make_llm: Callable[[], LLM],
    *,
    reset: bool = False,
    jira_site: str = JIRA_SITE,
    on_seeded: Callable[[Seeded], object] | None = None,
) -> list[Seeded]:
    """The snapshot's past meetings: those dated on or before its data_until."""
    return await seed_world(
        store,
        memory,
        make_llm,
        snapshot_meetings(name),
        reset=reset,
        jira_site=jira_site,
        on_seeded=on_seeded,
    )


async def seed_meeting(
    store: Store,
    memory: MeetingMemory,
    make_llm: Callable[[], LLM],
    team: Team,
    seed: SeedMeeting,
) -> Seeded:
    """Takes the meeting from wherever an earlier run left it to reviewed."""
    existing = await find_meeting(store, team.id, seed)
    result = Seeded(
        meeting_id="",
        day=seed.day,
        title=seed.title,
        outcome="seeded" if existing is None else "resumed",
    )
    if existing is not None and existing.status == "pushed":
        return result.model_copy(update={"meeting_id": existing.id, "outcome": "skipped"})
    host = seed.participant_ids[0]
    meeting = existing or await store.create_meeting(
        team.id,
        seed.title,
        host,
        scheduled_start=seed.started_at,
        duration_min=round((seed.ended_at - seed.started_at).total_seconds() / 60),
    )
    result.meeting_id = meeting.id
    if meeting.status == "scheduled":
        meeting = await store.start_meeting(meeting.id, seed.started_at)
    if meeting.status == "live":
        for member in team.member_ids:
            await store.add_participant(meeting.id, member)
        await store.add_segments(
            meeting.id, [s.model_copy(update={"meeting_id": meeting.id}) for s in seed.segments]
        )
        meeting = await store.transition_status(
            meeting.id, {"live"}, "processing", at=seed.ended_at
        )
    if meeting.status == "processing":
        log = ModelLog()
        pipeline = ReportPipeline(store, lambda: log.watch(make_llm()), memory)
        try:
            await pipeline.run(meeting.id)
        except Exception as e:
            progress = await store.report_progress(meeting.id)
            why = progress.error if progress and progress.error else f"{type(e).__name__}: {e}"
            raise SeedError(f"{describe(result)}: the write-up failed: {why}") from e
        result.models, result.cost = log.models, log.cost
        progress = await store.report_progress(meeting.id)
        if progress is None or progress.steps[INDEX] != REPORT_STEPS[INDEX]:
            step = progress.steps[INDEX] if progress else "no progress saved"
            raise SeedError(f"{describe(result)}: the write-up could not index it: {step}")
    elif meeting.status == "needs_review":
        await index_again(store, memory, meeting)
    # The host's review: the drafts are kept as written and nothing is pushed anywhere.
    await store.transition_status(meeting.id, {"needs_review"}, "pushed")
    return result


async def find_meeting(store: Store, tid: str, seed: SeedMeeting) -> Meeting | None:
    """The team's meeting with this title that started at this time, from an earlier run."""
    for meeting in await store.meetings(tid):
        when = meeting.started_at or meeting.scheduled_start
        if meeting.title == seed.title and when == seed.started_at:
            return meeting
    return None


async def index_again(store: Store, memory: MeetingMemory, meeting: Meeting) -> None:
    """For a meeting an earlier run wrote up but could not index: index it as the write-up does,
    then record the write-up as finished."""
    progress = await store.report_progress(meeting.id)
    if progress is not None and progress.done and progress.steps[INDEX] == REPORT_STEPS[INDEX]:
        return
    segments, report = await store.transcript(meeting.id), await store.report(meeting.id)
    await memory.index_meeting(meeting.team_id, meeting.id, segments, report)
    await store.save_report_progress(
        ReportProgress(
            meeting_id=meeting.id, steps=REPORT_STEPS, current=len(REPORT_STEPS), done=True
        )
    )


class ModelLog:
    """Wraps the write-up's LLM to note the models that answered and OpenRouter's cost."""

    def __init__(self) -> None:
        self.models: list[str] = []
        self.cost: float | None = None

    def watch(self, llm: LLM) -> LLM:
        return Watched(llm, self)

    def note(self, llm: LLM) -> None:
        if llm.last_model and llm.last_model not in self.models:
            self.models.append(llm.last_model)
        for provider in getattr(llm, "providers", (llm,)):
            usage = getattr(provider, "last_usage", None) if provider.last_model else None
            if isinstance(provider, OpenRouterLLM) and usage and "cost" in usage:
                self.cost = (self.cost or 0.0) + float(usage["cost"])


class Watched:
    def __init__(self, llm: LLM, log: ModelLog):
        self.llm = llm
        self.log = log

    @property
    def last_model(self) -> str | None:
        return self.llm.last_model

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        out = await self.llm.generate(prompt, system=system)
        self.log.note(self.llm)
        return out

    async def generate_structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        out = await self.llm.generate_structured(prompt, schema, system=system)
        self.log.note(self.llm)
        return out


class PacedEmbedder:
    """Keeps an embedder under a quota of `per_minute` texts in any minute, waiting as needed:
    Gemini's free tier counts each embedded text as a request. One call's texts are embedded in
    slices and joined, and must all come from one model."""

    def __init__(
        self,
        inner: Embedder,
        per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ):
        if per_minute < 1:
            raise ValueError("per_minute must be at least 1")
        self.inner = inner
        self.dim = inner.dim
        self.per_minute = per_minute
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[tuple[float, int]] = deque()

    async def embed(self, texts: list[str], *, task: EmbedTask = "document") -> Embeddings:
        if not texts:
            return await self.inner.embed(texts, task=task)
        parts: list[Embeddings] = []
        for start in range(0, len(texts), self.per_minute):
            batch = texts[start : start + self.per_minute]
            await self._wait_for(len(batch))
            parts.append(await self.inner.embed(batch, task=task))
        models = sorted({p.model for p in parts})
        if len(models) > 1:
            raise LLMError(f"Embeddings came from different models ({', '.join(models)})")
        return Embeddings(model=models[0], vectors=[v for p in parts for v in p.vectors])

    async def _wait_for(self, count: int) -> None:
        while True:
            now = self._clock()
            while self._sent and now - self._sent[0][0] >= 60:
                self._sent.popleft()
            if sum(n for _, n in self._sent) + count <= self.per_minute:
                break
            await self._sleep(60 - (now - self._sent[0][0]))
        self._sent.append((self._clock(), count))


def describe(result: Seeded) -> str:
    return f"{result.day.isoformat()} {result.title}"


def line(result: Seeded) -> str:
    if result.outcome == "skipped":
        return f"skipped: {describe(result)} (already seeded)"
    detail = f"model {', '.join(result.models)}" if result.models else "already written up"
    if result.cost is not None:
        detail += f", OpenRouter ${result.cost:.4f}"
    return f"{result.outcome}: {describe(result)} ({detail})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="world-seed",
        description="Seed the demo team, its settings and past meetings into DATABASE_URL.",
    )
    parser.add_argument(
        "--snapshot",
        choices=["dev", "demo"],
        default=Settings().world_snapshot,
        help="whose past meetings to seed (default: WORLD_SNAPSHOT, else dev)",
    )
    parser.add_argument(
        "--reset", action="store_true", help="remove the seeded team and everything under it first"
    )
    args = parser.parse_args(argv)

    settings = BrainSettings()
    if not settings.database_url:
        sys.exit("world-seed: DATABASE_URL is not configured")
    try:  # the real models or nothing: never a mock
        make_llm(settings)
        embedder = make_embedder(settings)
    except LLMUnavailable as e:
        sys.exit(f"world-seed: {e}")
    if embedder.dim != DIM:
        sys.exit(
            f"world-seed: GEMINI_EMBEDDING_DIM is {embedder.dim}, but meeting memory holds"
            f" {DIM}-dimension vectors"
        )
    meetings = snapshot_meetings(args.snapshot)
    if per_minute := Settings().world_seed_embeds_per_minute:
        embedder = PacedEmbedder(embedder, per_minute=per_minute)
    jira_site = settings.jira_base_url or JIRA_SITE

    async def run() -> int:
        dsn = settings.database_url
        assert dsn
        for name in await migrate(dsn):
            print(f"applied {name}")
        pool = await open_pool(dsn, max_size=2)
        try:
            store = PostgresStore(pool)
            memory = MeetingMemory(embedder, PgMemoryStore(pool))
            if args.reset:
                removed = await reset_team(store, memory)
                print(f"reset: removed {world_spec().company} and its {removed} meetings")
            results = await seed_world(
                store,
                memory,
                lambda: make_llm(settings),
                meetings,
                jira_site=jira_site,
                on_seeded=lambda result: print(line(result), flush=True),
            )
            tid = team_id(world_spec().company)
            decisions, tasks = await store.decisions(tid), await store.tasks(tid)
            superseded = sum(d.status == "superseded" for d in decisions)
            print(
                f"{world_spec().company} ({args.snapshot}): {len(results)} past meetings,"
                f" {len(decisions)} decisions ({superseded} superseded), {len(tasks)} task drafts"
            )
        except SeedError as e:
            print(f"world-seed: {e}", file=sys.stderr)
            return 1
        finally:
            await pool.close()
        return 0

    return asyncio.run(run())
