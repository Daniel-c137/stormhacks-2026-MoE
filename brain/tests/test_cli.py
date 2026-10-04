import asyncio
import io
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from brain.cli import main
from brain.llm import GeminiLLM
from brain.llm.mock import MockEmbedder
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.report import ProcessedMeeting
from brain.store import InMemoryStore
from contracts import Person, Team, TranscriptSegment

FIXTURES = Path(__file__).parent / "fixtures"


def test_report_writes_a_review_file_with_the_mock_llm(tmp_path, capsys):
    out = tmp_path / "review.json"

    code = main(
        [
            "report",
            str(FIXTURES / "standup.json"),
            "--mock-response",
            str(FIXTURES / "standup.extraction.json"),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    review = ProcessedMeeting.model_validate_json(out.read_text())
    assert review.meeting_id == "mtg-standup"
    assert [t.title for t in review.report.tasks] == [
        "Refund the 14 double-charged users",
        "Find an owner for the model retirement",
    ]
    assert [p.id for p in review.members] == ["p-alice", "p-bob", "p-carol"]
    printed = capsys.readouterr().out
    assert "MockLLM" in printed
    assert "2 task drafts" in printed


def test_report_names_the_model_that_answered(monkeypatch, tmp_path, capsys):
    extraction = (FIXTURES / "standup.extraction.json").read_text()

    class Models:
        async def generate_content(self, *, model, contents, config):
            if model == "busy-model":
                raise genai_errors.APIError(503, {"error": {"status": "UNAVAILABLE"}})
            return SimpleNamespace(text=extraction, parsed=None)

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    llm = GeminiLLM(models=["busy-model", "spare-model"], client=client)
    monkeypatch.setattr("brain.cli.make_llm", lambda: llm)

    code = main(["report", str(FIXTURES / "standup.json"), "--out", str(tmp_path / "r.json")])

    assert code == 0
    assert "from model spare-model" in capsys.readouterr().out


def test_report_without_gemini_config_says_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")
    # nor the OpenRouter fallback, which a developer's .env may set
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_MODELS", "")

    with pytest.raises(SystemExit) as exit_info:
        main(["report", str(FIXTURES / "standup.json"), "--out", str(tmp_path / "r.json")])

    assert "GEMINI_API_KEY" in str(exit_info.value.code)
    assert not (tmp_path / "r.json").exists()


def review_file(tmp_path) -> Path:
    out = tmp_path / "review.json"
    main(
        [
            "report",
            str(FIXTURES / "standup.json"),
            "--mock-response",
            str(FIXTURES / "standup.extraction.json"),
            "--out",
            str(out),
        ]
    )
    return out


def test_push_without_jira_config_says_what_is_missing(monkeypatch, tmp_path):
    for name in ("JIRA_MCP_URL", "JIRA_PROJECT_KEY", "JIRA_CLOUD_ID", "JIRA_BASE_URL"):
        monkeypatch.setenv(name, "")
    path = review_file(tmp_path)
    before = path.read_text()

    with pytest.raises(SystemExit) as exit_info:
        main(["push", str(path), "--all", "--approved-by", "Alice Moreau"])

    assert "JIRA_MCP_URL" in str(exit_info.value.code)
    assert path.read_text() == before


def test_push_needs_an_explicit_choice_of_drafts(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["push", str(review_file(tmp_path)), "--approved-by", "Alice Moreau"])

    assert exit_info.value.code == 2


def test_push_needs_a_named_approver(jira_env, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["push", str(review_file(tmp_path)), "--all", "--approved-by", " "])

    assert "approved" in str(exit_info.value.code)
    assert jira_env.created == []


def test_push_reports_failures_with_a_nonzero_exit(jira_env, tmp_path, capsys):
    path = review_file(tmp_path)
    review = ProcessedMeeting.model_validate_json(path.read_text())
    review.report.tasks[0].title = "FAIL on purpose"
    path.write_text(review.model_dump_json())

    assert main(["push", str(path), "--all", "--approved-by", "Alice Moreau"]) == 1

    out = capsys.readouterr().out
    assert "mtg-standup-task-1: failed" in out
    assert "mtg-standup-task-2: DS-117" in out


def test_push_prints_why_an_issue_was_created_unassigned(jira_env, tmp_path, capsys):
    jira_env.accounts = []
    path = review_file(tmp_path)

    assert main(["push", str(path), "--all", "--approved-by", "Alice Moreau"]) == 0

    out = capsys.readouterr().out
    assert "mtg-standup-task-1: DS-117" in out
    assert "mtg-standup-task-1: created unassigned:" in out


def test_migrate_without_database_url_says_what_is_missing(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(SystemExit) as exit_info:
        main(["migrate"])

    assert "DATABASE_URL" in str(exit_info.value.code)


def test_migrate_applies_the_migrations_once(monkeypatch, pg_dsn, capsys):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    assert main(["migrate"]) == 0
    first = capsys.readouterr().out
    assert main(["migrate"]) == 0
    second = capsys.readouterr().out

    assert "_core" in first
    assert "_core" not in second
    assert "up to date" in second


# purge-transcripts

ALEX = Person(id="u-alex", name="Alex Chen", short="Alex", initials="AC")
TEAM = Team(id="t-1", name="Checkout", member_ids=[ALEX.id])


def ended_with_transcript(store: InMemoryStore, title: str, days_ago: float):
    async def make():
        ended = datetime.now(UTC) - timedelta(days=days_ago)
        meeting = await store.create_meeting(TEAM.id, title, ALEX.id)
        await store.start_meeting(meeting.id, ended - timedelta(minutes=30))
        await store.add_segments(
            meeting.id,
            [
                TranscriptSegment(
                    seg_id=f"{meeting.id}-1",
                    meeting_id=meeting.id,
                    speaker_id=ALEX.id,
                    speaker_name=ALEX.name,
                    text="Refund the users",
                    is_final=True,
                    t_start=0,
                    t_end=2,
                )
            ],
        )
        await store.transition_status(meeting.id, {"live"}, "processing", at=ended)
        return await store.set_status(meeting.id, "needs_review")

    return asyncio.run(make())


@pytest.fixture
def retention_env(monkeypatch):
    """The CLI's store and memory, swapped for in-memory ones."""
    store = InMemoryStore(teams=[TEAM], people=[ALEX])
    env = SimpleNamespace(
        store=store, memory=MeetingMemory(MockEmbedder(dim=16), InMemoryMemoryStore())
    )

    @asynccontextmanager
    async def open_stores(settings):
        yield env.store, env.memory

    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("TRANSCRIPT_RETENTION_DAYS", "14")
    monkeypatch.setattr("brain.cli.open_retention_stores", open_stores)
    return env


def test_purge_dry_run_lists_what_would_go_and_deletes_nothing(retention_env, capsys):
    old = ended_with_transcript(retention_env.store, "Old standup", 20)
    recent = ended_with_transcript(retention_env.store, "Recent standup", 3)

    assert main(["purge-transcripts", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert old.id in out and "Old standup" in out
    assert recent.id not in out
    assert "dry run" in out.lower()
    assert len(asyncio.run(retention_env.store.transcript(old.id))) == 1
    assert asyncio.run(retention_env.store.meeting(old.id)).transcript_deleted_at is None


def test_purge_deletes_old_transcripts_and_prints_them(retention_env, capsys):
    old = ended_with_transcript(retention_env.store, "Old standup", 20)
    recent = ended_with_transcript(retention_env.store, "Recent standup", 3)

    assert main(["purge-transcripts"]) == 0

    out = capsys.readouterr().out
    assert old.id in out and "Old standup" in out
    assert recent.id not in out
    assert asyncio.run(retention_env.store.transcript(old.id)) == []
    assert asyncio.run(retention_env.store.meeting(old.id)).transcript_deleted_at is not None
    assert len(asyncio.run(retention_env.store.transcript(recent.id))) == 1

    assert main(["purge-transcripts"]) == 0
    assert "0 meetings" in capsys.readouterr().out


def test_purge_uses_the_configured_retention(retention_env, monkeypatch, capsys):
    old = ended_with_transcript(retention_env.store, "Old standup", 20)
    monkeypatch.setenv("TRANSCRIPT_RETENTION_DAYS", "30")

    assert main(["purge-transcripts"]) == 0

    assert old.id not in capsys.readouterr().out
    assert len(asyncio.run(retention_env.store.transcript(old.id))) == 1


def test_purge_says_when_memory_chunks_were_left(retention_env, capsys):
    ended_with_transcript(retention_env.store, "Old standup", 20)
    retention_env.memory = None

    assert main(["purge-transcripts"]) == 0

    assert "memory" in capsys.readouterr().out.lower()


def test_purge_refuses_a_retention_below_one_day(retention_env, monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_RETENTION_DAYS", "0")

    with pytest.raises(SystemExit) as exit_info:
        main(["purge-transcripts", "--dry-run"])

    assert "TRANSCRIPT_RETENTION_DAYS" in str(exit_info.value.code)


def test_purge_without_a_database_says_what_is_missing(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(SystemExit) as exit_info:
        main(["purge-transcripts", "--dry-run"])

    assert "DATABASE_URL is not configured" in str(exit_info.value.code)


# add-team and add-user: accounts are made from the command line; there is no public sign-up


@pytest.fixture
def db(monkeypatch, pg_dsn):
    """DATABASE_URL on a fresh migrated database, and a store to read it back with."""
    from brain.db import migrate
    from brain.pg_store import PostgresStore

    asyncio.run(migrate(pg_dsn))
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    return PostgresStore(pg_dsn)


def printed_password(out: str) -> str:
    """The one-time password add-user prints on its own line."""
    lines = [line for line in out.splitlines() if "password" in line.lower()]
    assert len(lines) == 1, out
    return lines[0].rsplit(" ", 1)[-1]


def add_alex(*extra: str) -> int:
    return main(
        [
            "add-user",
            "--team",
            "t-1",
            "--name",
            "Alex van der Berg",
            "--email",
            "Alex@Example.com",
            *extra,
        ]
    )


def test_add_team_creates_a_team(db, capsys):
    assert main(["add-team", "--id", "t-1", "--name", "Checkout"]) == 0

    team = asyncio.run(db.team("t-1"))
    assert (team.name, team.member_ids) == ("Checkout", [])
    assert "t-1" in capsys.readouterr().out


def test_add_team_again_renames_it_and_keeps_its_members(db, capsys):
    main(["add-team", "--id", "t-1", "--name", "Checkout"])
    add_alex()

    assert main(["add-team", "--id", "t-1", "--name", "Payments"]) == 0

    team = asyncio.run(db.team("t-1"))
    assert team.name == "Payments"
    assert len(team.member_ids) == 1


def test_add_user_creates_the_person_on_the_team_with_a_one_time_password(db, capsys):
    from brain.auth import verify_password

    main(["add-team", "--id", "t-1", "--name", "Checkout"])
    capsys.readouterr()

    assert add_alex() == 0

    out = capsys.readouterr().out
    password = printed_password(out)
    assert len(password) >= 16
    login = asyncio.run(db.login_by_email("alex@example.com"))
    assert verify_password(login.password_hash, password)
    assert login.password_hash not in out
    assert "argon2" not in out
    person = asyncio.run(db.person(login.person_id))
    assert (person.name, person.short, person.initials, person.email) == (
        "Alex van der Berg",
        "Alex",
        "AB",
        "Alex@Example.com",
    )
    assert asyncio.run(db.team("t-1")).member_ids == [person.id]
    assert asyncio.run(db.team_for_user(person.id)).id == "t-1"


def test_add_user_reads_the_password_from_stdin_and_prints_none(db, monkeypatch, capsys):
    from brain.auth import verify_password

    main(["add-team", "--id", "t-1", "--name", "Checkout"])
    capsys.readouterr()
    monkeypatch.setattr("sys.stdin", io.StringIO("a-chosen-password\n"))

    assert add_alex("--password-stdin") == 0

    out = capsys.readouterr().out
    assert "a-chosen-password" not in out
    assert "argon2" not in out
    login = asyncio.run(db.login_by_email("alex@example.com"))
    assert verify_password(login.password_hash, "a-chosen-password")


def test_add_user_refuses_a_short_password_from_stdin(db, monkeypatch):
    main(["add-team", "--id", "t-1", "--name", "Checkout"])
    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))

    with pytest.raises(SystemExit) as exit_info:
        add_alex("--password-stdin")

    assert "10" in str(exit_info.value.code)
    from brain.store import NotFound

    with pytest.raises(NotFound):
        asyncio.run(db.login_by_email("alex@example.com"))


def test_add_user_again_updates_the_same_person_and_resets_the_password(db, capsys):
    from brain.auth import verify_password

    main(["add-team", "--id", "t-1", "--name", "Checkout"])
    add_alex()
    first = asyncio.run(db.login_by_email("alex@example.com"))
    capsys.readouterr()

    code = main(["add-user", "--team", "t-1", "--name", "Alex Chen", "--email", "alex@example.com"])

    assert code == 0
    password = printed_password(capsys.readouterr().out)
    second = asyncio.run(db.login_by_email("alex@example.com"))
    assert second.person_id == first.person_id
    assert verify_password(second.password_hash, password)
    assert (asyncio.run(db.person(first.person_id))).name == "Alex Chen"
    assert asyncio.run(db.team("t-1")).member_ids == [first.person_id]


def test_add_user_needs_an_existing_team(db):
    with pytest.raises(SystemExit) as exit_info:
        add_alex()

    assert "t-1" in str(exit_info.value.code)
    assert "add-team" in str(exit_info.value.code)


@pytest.mark.parametrize("email", ["", "not-an-email", "two@@example.com"])
def test_add_user_refuses_an_invalid_email(db, email):
    main(["add-team", "--id", "t-1", "--name", "Checkout"])

    with pytest.raises(SystemExit) as exit_info:
        main(["add-user", "--team", "t-1", "--name", "Alex Chen", "--email", email])

    assert "email" in str(exit_info.value.code).lower()


@pytest.mark.parametrize(
    "argv",
    [
        ["add-team", "--id", "t-1", "--name", "Checkout"],
        ["add-user", "--team", "t-1", "--name", "Alex Chen", "--email", "alex@example.com"],
    ],
    ids=["add-team", "add-user"],
)
def test_account_commands_without_database_url_say_what_is_missing(monkeypatch, argv):
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert "DATABASE_URL" in str(exit_info.value.code)
