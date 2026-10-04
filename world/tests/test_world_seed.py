"""world-seed: the DropSubs team, its settings and its past meetings, written up by the brain's own
pipeline into the brain's store, in date order, once."""

import re
import sys
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path

import psycopg
import pytest

from brain.agent.pipeline import REPORT_STEPS
from brain.config import Settings as BrainSettings
from brain.db import migrate, open_pool
from brain.llm import LLMError, MockEmbedder, MockLLM, make_embedder, make_llm
from brain.memory import InMemoryMemoryStore, MeetingMemory, PgMemoryStore
from brain.pg_store import PostgresStore
from brain.report import ExtractedDecision, ExtractedTask, ReportExtraction
from brain.report.decisions import DecisionVerdict, DecisionVerdicts
from brain.store import InMemoryStore, Store
from contracts import DecisionRelation, Person
from world import seed
from world.config import MOCK_DATA_DIR
from world.past_meetings import MeetingFileError, load_meeting, load_meetings, snapshot_meetings
from world.seed import PacedEmbedder, SeedError, seed_people, seed_world

# The brain's embedded Postgres fixtures (pgserver), shared rather than copied. Imported here, not
# from a conftest: a second module named conftest would shadow the brain tests' own.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain" / "tests"))
from pg_support import fresh_database, pg_server  # noqa: F401  (pg_server is a fixture)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg_dsn(pg_server):  # noqa: F811
    with fresh_database(pg_server) as dsn:
        yield dsn


MEETINGS_DIR = MOCK_DATA_DIR / "meetings"
DAYS = [date(2026, 9, 23), date(2026, 9, 28), date(2026, 9, 30), date(2026, 10, 2)]
TITLES = [
    "DropSubs Standup and sync",
    "DropSubs Sprint 8 planning",
    "DropSubs Standup and sync",
    "DropSubs Standup and sync",
]
PEOPLE = ["p-danial", "p-hossein", "p-mohammad-reza", "p-reyhaneh"]
TEAM = "team-dropsubs"


# parsing the meeting files


def test_the_four_past_meetings_parse_into_final_segments():
    meetings = load_meetings(MEETINGS_DIR)

    assert [m.day for m in meetings] == DAYS
    assert [m.title for m in meetings] == TITLES
    assert [m.id for m in meetings] == [f"seed-{d.isoformat()}" for d in DAYS]
    for meeting in meetings:
        assert meeting.participant_ids == PEOPLE
        assert meeting.timezone == "America/Vancouver"
        speakers = {(s.speaker_id, s.speaker_name) for s in meeting.segments}
        assert speakers == {
            ("p-danial", "Danial"),
            ("p-hossein", "Hossein"),
            ("p-mohammad-reza", "Mohammad Reza"),
            ("p-reyhaneh", "Reyhaneh"),
        }
        segments = meeting.segments
        assert len(segments) > 250
        assert all(s.is_final and s.meeting_id == meeting.id for s in segments)
        assert len({s.seg_id for s in segments}) == len(segments)
        assert all(a.t_start <= b.t_start for a, b in pairwise(segments))
        assert all(a.t_end == b.t_start for a, b in pairwise(segments))
        assert 0 < segments[-1].t_end - segments[-1].t_start <= 10
        assert meeting.ended_at > meeting.started_at


def test_a_meeting_keeps_its_real_times_and_lines():
    first = load_meeting(MEETINGS_DIR / "2026-09-23.md")

    assert first.started_at == datetime(2026, 9, 23, 17, 0, tzinfo=UTC)
    assert first.ended_at == datetime(2026, 9, 23, 17, 59, tzinfo=UTC)
    assert [p.model_dump() for p in first.participants][:2] == [
        {"name": "Danial", "role": "CEO"},
        {"name": "Hossein", "role": "AI engineer"},
    ]
    opening = first.segments[0]
    assert (opening.speaker_id, opening.text) == ("p-danial", "Morning. Can everyone hear me?")
    assert (opening.t_start, opening.t_end) == (3.0, 6.0)  # [00:00:03], next line [00:00:06]


FRONT = """---
meeting_id: seed-2026-09-01
title: "Tiny sync"
date: 2026-09-01
start: 2026-09-01T10:00:00-07:00
end: 2026-09-01T10:30:00-07:00
timezone: America/Vancouver
participants:
  - name: Danial
    role: CEO
  - name: Mohammad Reza
    role: Backend engineer
is_seed: true
---
"""


def write(directory: Path, name: str, body: str, front: str = FRONT) -> Path:
    path = directory / name
    path.write_text(front + body)
    return path


def test_headings_notes_and_blank_lines_are_skipped(tmp_path):
    path = write(
        tmp_path,
        "2026-09-01.md",
        "\n# Tiny sync\n\nAttendees: Danial (CEO), Mohammad Reza (Backend engineer)\n\n"
        "[00:00:01] Danial: Hi.\n[Recording paused]\n\n"
        "[00:01:05] Mohammad Reza: Hello: there.\n",
    )

    meeting = load_meeting(path)

    assert [(s.speaker_id, s.text, s.t_start, s.t_end) for s in meeting.segments] == [
        ("p-danial", "Hi.", 1.0, 65.0),
        ("p-mohammad-reza", "Hello: there.", 65.0, meeting.segments[-1].t_end),
    ]
    assert meeting.segments[-1].t_end > 65.0


def test_an_unknown_speaker_names_the_file_and_line(tmp_path):
    path = write(tmp_path, "2026-09-01.md", "\n[00:00:01] Danial: Hi.\n[00:00:04] Sara: Hey.\n")

    with pytest.raises(MeetingFileError, match=r"2026-09-01\.md:17: .*Sara"):
        load_meeting(path)


@pytest.mark.parametrize(
    "line",
    [
        "Danial: no timestamp",
        "[0:00:04] Danial: short hour",
        "[00:00:04] Danial:",
        "[00:61:04] Danial: no such minute",
        "[00:00:04]Danial: no space",
        "just some words",
    ],
)
def test_a_malformed_line_names_the_file_and_line(tmp_path, line):
    path = write(tmp_path, "2026-09-01.md", f"\n[00:00:01] Danial: Hi.\n{line}\n")

    with pytest.raises(MeetingFileError, match=r"2026-09-01\.md:17: "):
        load_meeting(path)


def test_a_line_that_goes_back_in_time_is_refused(tmp_path):
    path = write(tmp_path, "2026-09-01.md", "\n[00:00:09] Danial: Hi.\n[00:00:04] Danial: Ho.\n")

    with pytest.raises(MeetingFileError, match=r"2026-09-01\.md:17: "):
        load_meeting(path)


def test_bad_front_matter_names_the_file(tmp_path):
    path = write(
        tmp_path, "2026-09-01.md", "[00:00:01] Danial: Hi.\n", front="---\ntitle: x\n---\n"
    )

    with pytest.raises(MeetingFileError, match=r"2026-09-01\.md"):
        load_meeting(path)


def test_a_snapshot_has_the_meetings_up_to_its_data_until_in_date_order():
    assert [m.day for m in snapshot_meetings("demo", MEETINGS_DIR)] == DAYS
    assert snapshot_meetings("dev", MEETINGS_DIR) == []  # dev's data stops on 2026-08-28


# the people and the team


def test_the_seeded_people_come_from_the_meetings():
    people = seed_people(load_meetings(MEETINGS_DIR))

    assert people == [
        Person(
            id="p-danial",
            name="Danial",
            short="Danial",
            initials="D",
            title="CEO",
            email="danial@dropsubs.example",
        ),
        Person(
            id="p-hossein",
            name="Hossein",
            short="Hossein",
            initials="H",
            title="AI engineer",
            email="hossein@dropsubs.example",
        ),
        Person(
            id="p-mohammad-reza",
            name="Mohammad Reza",
            short="Mohammad",
            initials="MR",
            title="Backend engineer",
            email="mohammadreza@dropsubs.example",
        ),
        Person(
            id="p-reyhaneh",
            name="Reyhaneh",
            short="Reyhaneh",
            initials="R",
            title="Frontend engineer",
            email="reyhaneh@dropsubs.example",
        ),
    ]
    assert seed_people() == people  # by default, from every meeting file


# seeding, with scripted models


def extraction(prompt: str) -> ReportExtraction:
    """One decision and one task per meeting, named after its date."""
    match = re.search(r"^Date: (\d{4}-\d{2}-\d{2})", prompt, re.MULTILINE)
    assert match, "the prompt dates the meeting"
    day = match[1]
    return ReportExtraction(
        summary=f"The team met on {day}.",
        decisions=[
            ExtractedDecision(
                text=f"Charge the fee agreed on {day}", made_by_id="p-danial", evidence=["s1"]
            )
        ],
        tasks=[ExtractedTask(title=f"Follow up on {day}", owner_id="p-reyhaneh", evidence=["s2"])],
    )


def supersede(prompt: str) -> DecisionVerdicts:
    """Every new decision supersedes the first past decision it is shown."""
    new = re.search(r"New decision:\n\[([^\]]+)\]", prompt)
    past = re.search(r"Past decisions to compare with:\n  \[([^\]]+)\]", prompt)
    assert new and past
    return DecisionVerdicts(
        verdicts=[
            DecisionVerdict(
                decision_id=new[1],
                verdict="supersedes",
                past_decision_id=past[1],
                reason="It replaces the earlier fee.",
                confidence=0.9,
            )
        ]
    )


def scripted() -> MockLLM:
    return MockLLM(structured={ReportExtraction: extraction, DecisionVerdicts: supersede})


def in_memory() -> tuple[InMemoryStore, MeetingMemory]:
    return InMemoryStore(), MeetingMemory(MockEmbedder(), InMemoryMemoryStore(dim=768))


DEMO = snapshot_meetings("demo", MEETINGS_DIR)


def prompted_days(llm: MockLLM) -> list[str]:
    return [
        m[1]
        for call in llm.calls
        if call.schema is ReportExtraction
        and (m := re.search(r"^Date: (\d{4}-\d{2}-\d{2})", call.prompt, re.MULTILINE))
    ]


async def chunks_by_meeting(memory: MeetingMemory) -> dict[str, int]:
    hits = await memory.search(TEAM, "fee team met follow up morning", k=100_000)
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.chunk.meeting_id] = counts.get(hit.chunk.meeting_id, 0) + 1
    return counts


async def assert_seeded(store: Store, memory: MeetingMemory) -> list[str]:
    """The finished demo world; returns the meeting ids, oldest first."""
    team = await store.team(TEAM)
    assert team.name == "DropSubs"
    assert team.member_ids == PEOPLE
    assert (team.github_repo, team.jira_project) == ("dropsubs/dropsubs", "DS")
    assert [p.id for p in await store.members(TEAM)] == PEOPLE

    settings = await store.settings(TEAM)
    assert settings.github.repo == "dropsubs/dropsubs"
    assert (settings.jira.site, settings.jira.project) == ("https://dropsubs.atlassian.net", "DS")
    assert settings.timezone == "America/Vancouver"
    assert (settings.sensitivity, settings.interrupt_minutes) == ("balanced", 5)
    assert not settings.github.connected and not settings.jira.connected

    meetings = list(reversed(await store.meetings(TEAM)))
    assert [m.title for m in meetings] == TITLES
    for meeting, parsed in zip(meetings, DEMO, strict=True):
        assert meeting.status == "pushed"
        assert (meeting.started_at, meeting.ended_at) == (parsed.started_at, parsed.ended_at)
        assert meeting.host_id == "p-danial"
        assert meeting.participant_ids == PEOPLE
        assert meeting.jira_keys == []
        transcript = await store.transcript(meeting.id)
        assert [(s.speaker_id, s.text, s.t_start) for s in transcript] == [
            (s.speaker_id, s.text, s.t_start) for s in parsed.segments
        ]
        progress = await store.report_progress(meeting.id)
        assert progress is not None and progress.done and progress.error is None
        assert progress.steps == REPORT_STEPS
        report = await store.report(meeting.id)
        assert report.summary == f"The team met on {parsed.day.isoformat()}."

    ids = [m.id for m in meetings]
    # In date order, each meeting's decision retires the one before it.
    decisions = {d.meeting_id: d for d in await store.decisions(TEAM)}
    for earlier, later in pairwise(ids):
        assert decisions[earlier].status == "superseded"
        assert decisions[earlier].relation == DecisionRelation(
            type="superseded_by", decision_id=decisions[later].id
        )
    assert decisions[ids[-1]].status == "active"
    # Reviewed, never pushed: drafts kept for the team, with no Jira key.
    tasks = await store.tasks(TEAM)
    assert len(tasks) == 4
    assert all(
        t.include and t.key is None and t.jira_status == "draft" and t.owner_id == "p-reyhaneh"
        for t in tasks
    )
    counts = await chunks_by_meeting(memory)
    assert set(counts) == set(ids)
    return ids


async def test_seeding_writes_up_every_meeting_in_date_order():
    store, memory = in_memory()
    llm = scripted()

    seeded = await seed_world(store, memory, lambda: llm, DEMO)

    assert [(s.outcome, s.day, s.title, s.models) for s in seeded] == [
        ("seeded", d, t, ["mock"]) for d, t in zip(DAYS, TITLES, strict=True)
    ]
    assert prompted_days(llm) == [d.isoformat() for d in DAYS]
    ids = await assert_seeded(store, memory)
    assert [s.meeting_id for s in seeded] == ids


async def test_each_meeting_is_reported_as_it_is_seeded():
    store, memory = in_memory()
    reported = []

    await seed_world(store, memory, scripted, DEMO[:2], on_seeded=reported.append)

    assert [(s.outcome, s.day) for s in reported] == [("seeded", DAYS[0]), ("seeded", DAYS[1])]


async def test_seeding_again_changes_nothing():
    store, memory = in_memory()
    await seed_world(store, memory, scripted, DEMO)
    meetings, decisions, tasks = (
        await store.meetings(TEAM),
        await store.decisions(TEAM),
        await store.tasks(TEAM),
    )
    chunks = await chunks_by_meeting(memory)
    llm = scripted()

    again = await seed_world(store, memory, lambda: llm, DEMO)

    assert [(s.outcome, s.models) for s in again] == [("skipped", [])] * 4
    assert llm.calls == []
    assert await store.meetings(TEAM) == meetings
    assert await store.decisions(TEAM) == decisions
    assert await store.tasks(TEAM) == tasks
    assert await chunks_by_meeting(memory) == chunks


async def test_seeding_keeps_settings_changed_since():
    store, memory = in_memory()
    await seed_world(store, memory, scripted, DEMO[:1])
    changed = (await store.settings(TEAM)).model_copy(update={"voice": "voice-1"})
    await store.save_settings(changed)

    await seed_world(store, memory, scripted, DEMO[:1])

    assert await store.settings(TEAM) == changed


async def test_reset_removes_the_seeded_team_first():
    store, memory = in_memory()
    await seed_world(store, memory, scripted, DEMO)
    old = {m.id for m in await store.meetings(TEAM)}
    extra = await store.create_meeting(TEAM, "Not from the world", "p-danial")

    reseeded = await seed_world(store, memory, scripted, DEMO, reset=True)

    assert [s.outcome for s in reseeded] == ["seeded"] * 4
    ids = await assert_seeded(store, memory)
    assert not old & set(ids)
    assert extra.id not in {m.id for m in await store.meetings(TEAM)}
    assert set(await chunks_by_meeting(memory)) == set(ids)  # no chunks of deleted meetings


async def test_a_failed_write_up_stops_the_seed_and_the_next_run_finishes_it():
    store, memory = in_memory()
    made: list[MockLLM] = []

    def flaky() -> MockLLM:
        # One model per write-up; the second meeting's has nothing scripted, so it fails.
        made.append(MockLLM() if len(made) == 1 else scripted())
        return made[-1]

    with pytest.raises(SeedError, match="2026-09-28"):
        await seed_world(store, memory, flaky, DEMO)
    first, second = reversed(await store.meetings(TEAM))
    assert (first.status, second.status) == ("pushed", "processing")
    progress = await store.report_progress(second.id)
    assert progress is not None and progress.error

    resumed = await seed_world(store, memory, scripted, DEMO)

    assert [s.outcome for s in resumed] == ["skipped", "resumed", "seeded", "seeded"]
    await assert_seeded(store, memory)


async def test_a_failed_index_stops_the_seed_and_the_next_run_indexes_it():
    store, memory = in_memory()

    class FailingEmbedder(MockEmbedder):
        async def embed(self, texts, *, task="document"):
            raise LLMError("embedding model down")

    broken = MeetingMemory(FailingEmbedder(), memory.store)

    with pytest.raises(SeedError, match=r"2026-09-23.*index"):
        await seed_world(store, broken, scripted, DEMO[:1])
    (meeting,) = await store.meetings(TEAM)
    assert meeting.status == "needs_review"
    llm = scripted()

    resumed = await seed_world(store, memory, lambda: llm, DEMO[:1])

    assert [s.outcome for s in resumed] == ["resumed"]
    assert llm.calls == []  # the report was kept; only memory was redone
    assert (await store.meeting(meeting.id)).status == "pushed"
    assert set(await chunks_by_meeting(memory)) == {meeting.id}


async def test_seeding_into_postgres(pg_dsn):
    await migrate(pg_dsn)
    pool = await open_pool(pg_dsn, max_size=4)
    try:
        store = PostgresStore(pool)
        memory = MeetingMemory(MockEmbedder(), PgMemoryStore(pool))

        seeded = await seed_world(store, memory, scripted, DEMO)
        assert [s.outcome for s in seeded] == ["seeded"] * 4
        ids = await assert_seeded(store, memory)

        again = await seed_world(store, memory, scripted, DEMO)
        assert [s.outcome for s in again] == ["skipped"] * 4
        assert await assert_seeded(store, memory) == ids

        await seed_world(store, memory, scripted, DEMO, reset=True)
        new_ids = await assert_seeded(store, memory)
        assert not set(ids) & set(new_ids)
        with psycopg.connect(pg_dsn) as conn:
            rows = conn.execute("select distinct meeting_id from memory_chunks").fetchall()
        assert {r[0] for r in rows} == set(new_ids)
    finally:
        await pool.close()


# the command


@pytest.fixture
def mock_models(monkeypatch):
    """world-seed's model factories, scripted; the configuration checks still run."""
    monkeypatch.setattr(seed, "make_llm", lambda settings: scripted())
    monkeypatch.setattr(seed, "make_embedder", lambda settings: MockEmbedder())
    for name, value in {
        "GEMINI_API_KEY": "test-key",
        "GEMINI_MODEL": "test-model",
        "GEMINI_EMBEDDING_MODEL": "test-embedding",
        "GEMINI_EMBEDDING_DIM": "768",
    }.items():
        monkeypatch.setenv(name, value)


def test_the_command_migrates_then_seeds_and_says_what_it_did(
    pg_dsn, monkeypatch, capsys, mock_models
):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    lines = [f"{d.isoformat()} {t}" for d, t in zip(DAYS, TITLES, strict=True)]

    assert seed.main(["--snapshot", "demo"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert any(line.startswith("applied ") for line in out)
    assert [line for line in out if not line.startswith("applied ")][:4] == [
        f"seeded: {line} (model mock)" for line in lines
    ]

    assert seed.main(["--snapshot", "demo"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:4] == [f"skipped: {line} (already seeded)" for line in lines]

    assert seed.main(["--snapshot", "demo", "--reset"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("reset: removed DropSubs")
    assert out[1:5] == [f"seeded: {line} (model mock)" for line in lines]


def test_the_command_needs_a_database(monkeypatch, mock_models):
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(SystemExit, match="DATABASE_URL"):
        seed.main(["--snapshot", "demo"])


def test_the_command_refuses_to_run_without_a_language_model(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@unreachable.invalid/none")
    for name in ("GEMINI_MODEL", "OPENROUTER_MODELS"):
        monkeypatch.setenv(name, "")

    with pytest.raises(SystemExit, match="Gemini is not configured"):
        seed.main(["--snapshot", "demo"])


def test_the_command_refuses_to_run_without_embeddings(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@unreachable.invalid/none")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "")

    with pytest.raises(SystemExit, match="GEMINI_EMBEDDING_MODEL"):
        seed.main(["--snapshot", "demo"])


def test_the_command_refuses_embeddings_the_database_cannot_hold(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@unreachable.invalid/none")
    for name, value in {
        "GEMINI_API_KEY": "test-key",
        "GEMINI_MODEL": "test-model",
        "GEMINI_EMBEDDING_MODEL": "test-embedding",
        "GEMINI_EMBEDDING_DIM": "1536",
    }.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit, match="1536"):
        seed.main(["--snapshot", "demo"])


# against the real models, once


settings = BrainSettings()


@pytest.mark.live
@pytest.mark.skipif(
    not (
        (settings.gemini_api_key and settings.gemini_model)
        or (settings.openrouter_api_key and settings.openrouter_models)
    )
    or not (
        settings.gemini_api_key
        and settings.gemini_embedding_model
        and settings.gemini_embedding_dim
    ),
    reason="set GEMINI_API_KEY, GEMINI_MODEL (or OPENROUTER_*) and GEMINI_EMBEDDING_* to run",
)
async def test_the_earliest_meeting_is_written_up_by_the_real_models():
    store = InMemoryStore()
    # Gemini's free tier counts each embedded text against 100 a minute; a meeting has ~350.
    embedder = PacedEmbedder(make_embedder(settings), per_minute=90)
    memory = MeetingMemory(embedder, InMemoryMemoryStore(dim=settings.gemini_embedding_dim))

    (seeded,) = await seed_world(store, memory, lambda: make_llm(settings), DEMO[:1])
    print(f"answered by {', '.join(seeded.models)}; OpenRouter cost {seeded.cost}")

    assert seeded.outcome == "seeded" and seeded.models
    meeting = await store.meeting(seeded.meeting_id)
    assert meeting.status == "pushed"
    report = await store.report(meeting.id)
    assert report.summary and report.decisions and report.tasks
    assert all(t.key is None and t.include for t in report.tasks)
    members = {p.id for p in await store.members(TEAM)}
    assert all(t.owner_id is None or t.owner_id in members for t in report.tasks)
    hits = await memory.search(TEAM, "iPhone app App Store submission", k=5)
    assert hits and all(h.chunk.meeting_id == meeting.id for h in hits)


# pacing embeddings to a per-minute quota


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class CountingEmbedder(MockEmbedder):
    def __init__(self, clock: Clock, models: list[str] | None = None):
        super().__init__()
        self.clock = clock
        self.batches: list[tuple[float, int]] = []
        self.models = models

    async def embed(self, texts, *, task="document"):
        self.batches.append((self.clock.now, len(texts)))
        out = await super().embed(texts, task=task)
        if self.models:
            out = out.model_copy(update={"model": self.models[len(self.batches) - 1]})
        return out


async def test_paced_embeddings_never_send_more_texts_in_a_minute_than_allowed():
    clock = Clock()
    inner = CountingEmbedder(clock)
    paced = PacedEmbedder(inner, per_minute=100, clock=clock, sleep=clock.sleep)
    texts = [f"text {i}" for i in range(250)]

    embedded = await paced.embed(texts)
    again = await paced.embed(["one more"], task="query")

    assert embedded == await MockEmbedder().embed(texts)  # same vectors, in order, one model
    assert again.vectors == (await MockEmbedder().embed(["one more"])).vectors
    assert inner.batches == [(1000.0, 100), (1060.0, 100), (1120.0, 50), (1180.0, 1)]
    assert paced.dim == inner.dim
    assert inner.calls[-1] == (["one more"], "query")


async def test_paced_embeddings_from_two_models_are_refused():
    clock = Clock()
    inner = CountingEmbedder(clock, models=["a", "b"])
    paced = PacedEmbedder(inner, per_minute=2, clock=clock, sleep=clock.sleep)

    with pytest.raises(LLMError, match="different models"):
        await paced.embed(["x", "y", "z"])


def test_the_command_paces_embeddings_when_asked(pg_dsn, monkeypatch, capsys, mock_models):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    monkeypatch.setenv("WORLD_SEED_EMBEDS_PER_MINUTE", "100000")
    made: list[PacedEmbedder] = []

    def paced(inner, per_minute):
        made.append(PacedEmbedder(inner, per_minute=per_minute))
        return made[-1]

    monkeypatch.setattr(seed, "PacedEmbedder", paced)

    assert seed.main(["--snapshot", "demo"]) == 0

    assert [p.per_minute for p in made] == [100000]
    assert "seeded: 2026-09-23" in capsys.readouterr().out
