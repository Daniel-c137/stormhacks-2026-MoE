"""`brain report` turns a transcript into a review file of the report and task drafts.
`brain push` creates Jira issues for the drafts a named person approved.
`brain migrate` applies db/migrations to DATABASE_URL.
`brain add-team` and `brain add-user` make teams and accounts; sign-up is invite-only (#143).
`brain set-admin` grants or revokes admin: only an admin changes team settings and accounts.
`brain purge-transcripts` deletes transcripts older than TRANSCRIPT_RETENTION_DAYS."""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from psycopg import errors as pg_errors
from pydantic import ValidationError

from brain.accounts import (
    InvalidAccount,
    LastAdmin,
    clean_email,
    clean_name,
    existing_person,
    generate_password,
    save_account,
    set_admin,
)
from brain.auth import MAX_PASSWORD, MIN_PASSWORD, hash_password
from brain.config import Settings
from brain.db import migrate
from brain.jira import ApprovalRequired, JiraPusher, JiraUnavailable, apply_results, jira_config
from brain.llm import LLM, LLMError, MockLLM, make_llm
from brain.pg_store import PostgresStore
from brain.report import ProcessedMeeting, ReportExtraction, build_report, load_transcript
from brain.retention import (
    RetentionUnavailable,
    open_retention_stores,
    purge_transcripts,
    retention_cutoff,
    transcripts_due,
)
from brain.store import Conflict, NotFound, Store
from contracts import Meeting, TaskPushRequest, Team


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brain")
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="write the report and task drafts for review")
    report.add_argument(
        "transcript", type=Path, help="TranscriptInput .json, or text lines '[mm:ss] Name: text'"
    )
    report.add_argument(
        "--date",
        type=datetime.fromisoformat,
        help="meeting start for text transcripts, e.g. 2026-10-02T09:30",
    )
    report.add_argument("--out", type=Path, help="review file (default: <transcript>.review.json)")
    report.add_argument(
        "--mock-response",
        type=Path,
        metavar="EXTRACTION_JSON",
        help="use MockLLM with this scripted ReportExtraction instead of Gemini",
    )
    report.set_defaults(run=run_report)

    push = commands.add_parser("push", help="create Jira issues for drafts a person approved")
    push.add_argument("review", type=Path, help="review file written by 'brain report'")
    which = push.add_mutually_exclusive_group(required=True)
    which.add_argument("--task", action="append", dest="task_ids", metavar="TASK_ID")
    which.add_argument("--all", action="store_true", help="every draft still marked include")
    push.add_argument("--approved-by", required=True, metavar="NAME")
    push.set_defaults(run=run_push)

    migrate_cmd = commands.add_parser("migrate", help="apply db/migrations to DATABASE_URL")
    migrate_cmd.set_defaults(run=run_migrate)

    add_team = commands.add_parser("add-team", help="create a team, or rename one")
    add_team.add_argument("--id", required=True, dest="team_id")
    add_team.add_argument("--name", required=True)
    add_team.set_defaults(run=run_add_team)

    add_user = commands.add_parser(
        "add-user", help="add a person to a team with an email and password login"
    )
    add_user.add_argument("--team", required=True, dest="team_id")
    add_user.add_argument("--name", required=True, help='full name, e.g. "Alex Chen"')
    add_user.add_argument("--email", required=True)
    add_user.add_argument(
        "--password-stdin",
        action="store_true",
        help="read the password from stdin instead of generating one",
    )
    add_user.add_argument(
        "--admin", action="store_true", help="make them an admin (an existing admin stays one)"
    )
    add_user.set_defaults(run=run_add_user)

    admin = commands.add_parser(
        "set-admin", help="make the person who signs in with an email an admin, or revoke it"
    )
    admin.add_argument("--email", required=True)
    admin.add_argument("--revoke", action="store_true", help="revoke admin instead")
    admin.set_defaults(run=run_set_admin)

    purge = commands.add_parser(
        "purge-transcripts", help="delete transcripts older than TRANSCRIPT_RETENTION_DAYS"
    )
    purge.add_argument("--dry-run", action="store_true", help="list them, delete nothing")
    purge.set_defaults(run=run_purge_transcripts)

    args = parser.parse_args(argv)
    return args.run(args)


def run_report(args: argparse.Namespace) -> int:
    meeting = load_transcript(args.transcript, started_at=args.date)
    llm: LLM
    source: str | None
    if args.mock_response:
        scripted = ReportExtraction.model_validate_json(args.mock_response.read_text())
        llm = MockLLM(structured={ReportExtraction: scripted})
        source = "MockLLM (scripted, not Gemini)"
    else:
        try:
            llm = make_llm()
        except LLMError as e:
            sys.exit(f"brain report: {e}")
        source = None

    try:
        report = asyncio.run(build_report(llm, meeting))
    except (LLMError, ValueError) as e:
        sys.exit(f"brain report: {e}")
    source = source or f"model {llm.last_model}"

    out = args.out or args.transcript.with_suffix(".review.json")
    review = ProcessedMeeting(
        meeting_id=meeting.meeting_id,
        title=meeting.title,
        started_at=meeting.started_at,
        members=meeting.people(),
        report=report,
    )
    out.write_text(review.model_dump_json(indent=2) + "\n")
    print(f"{len(report.tasks)} task drafts, {len(report.decisions)} decisions from {source}")
    print(f"Review and edit {out} before pushing anything.")
    return 0


def run_push(args: argparse.Namespace) -> int:
    review = ProcessedMeeting.model_validate_json(args.review.read_text())
    try:
        config = jira_config(Settings())
    except JiraUnavailable as e:
        sys.exit(f"brain push: {e}")
    request = TaskPushRequest(
        task_ids=args.task_ids or [t.id for t in review.report.tasks if t.include],
        destination="jira",
        approved_by=args.approved_by,
    )
    try:
        results = asyncio.run(JiraPusher(config).push(review, request))
    except ApprovalRequired as e:
        sys.exit(f"brain push: {e}")

    review.report.tasks = apply_results(review.report.tasks, results)
    args.review.write_text(review.model_dump_json(indent=2) + "\n")
    for r in results:
        print(
            f"{r.task_id}: {r.key} {r.url or ''}".rstrip()
            if r.key
            else f"{r.task_id}: failed: {r.error}"
        )
        if r.warning:
            print(f"{r.task_id}: {r.warning}")
    return 0 if all(r.key for r in results) else 1


def run_migrate(args: argparse.Namespace) -> int:
    dsn = Settings().database_url
    if not dsn:
        sys.exit("brain migrate: DATABASE_URL is not configured")
    applied = asyncio.run(migrate(dsn))
    for name in applied:
        print(f"applied {name}")
    if not applied:
        print("Database is up to date.")
    return 0


def account_store(command: str) -> Store:
    dsn = Settings().database_url
    if not dsn:
        sys.exit(f"brain {command}: DATABASE_URL is not configured")
    return PostgresStore(dsn)


def run_account_command(command: str, work) -> int:
    try:
        return asyncio.run(work)
    except pg_errors.UndefinedTable:
        sys.exit(f"brain {command}: the database has no tables yet; run `brain migrate` first")


def run_add_team(args: argparse.Namespace) -> int:
    store = account_store("add-team")
    name = " ".join(args.name.split())
    if not args.team_id.strip() or not name:
        sys.exit("brain add-team: --id and --name must not be blank")

    async def add() -> int:
        try:
            team = await store.team(args.team_id)
        except NotFound:
            team = Team(id=args.team_id, name=name, member_ids=[])
        team = await store.create_team(team.model_copy(update={"name": name}))
        print(f"Team {team.id} ({team.name}), {len(team.member_ids)} members.")
        return 0

    return run_account_command("add-team", add())


def run_add_user(args: argparse.Namespace) -> int:
    store = account_store("add-user")
    try:
        name = clean_name(args.name)
        email = clean_email(args.email)
    except InvalidAccount as e:
        sys.exit(f"brain add-user: {e}")
    if args.password_stdin:
        password = sys.stdin.read().rstrip("\r\n")
        if not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
            sys.exit(
                f"brain add-user: the password must be {MIN_PASSWORD} to {MAX_PASSWORD} characters"
            )
        generated = None
    else:
        password = generated = generate_password()

    async def add() -> int:
        try:
            team = await store.team(args.team_id)
        except NotFound:
            sys.exit(
                f"brain add-user: there is no team {args.team_id};"
                f" create it with `brain add-team --id {args.team_id} --name ...`"
            )
        try:
            person = await save_account(
                store,
                team,
                name=name,
                email=email,
                password_hash=hash_password(password),
                person=await existing_person(store, team, email),
                is_admin=True if args.admin else None,
            )
        except Conflict:
            sys.exit(f"brain add-user: another person already signs in with {email}")
        role = " as an admin" if person.is_admin else ""
        print(f"{person.name} ({person.id}) is on team {team.id}{role} and signs in as {email}.")
        if generated:
            print(f"One-time password, shown only now: {generated}")
        return 0

    return run_account_command("add-user", add())


def run_set_admin(args: argparse.Namespace) -> int:
    store = account_store("set-admin")
    email = args.email.strip()

    async def change() -> int:
        try:
            login = await store.login_by_email(email)
            person = await set_admin(store, login.person_id, not args.revoke)
        except NotFound:
            sys.exit(f"brain set-admin: no one signs in with {email}")
        except LastAdmin as e:
            sys.exit(f"brain set-admin: {e}; make someone else an admin first")
        print(f"{person.name} ({email}) is {'an' if person.is_admin else 'not an'} admin.")
        return 0

    return run_account_command("set-admin", change())


def run_purge_transcripts(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as e:
        sys.exit(f"brain purge-transcripts: TRANSCRIPT_RETENTION_DAYS must be at least 1 ({e})")
    days = settings.transcript_retention_days
    now = datetime.now(UTC)
    cutoff = retention_cutoff(now, days)

    async def run():
        async with open_retention_stores(settings) as (store, memory):
            if args.dry_run:
                return await transcripts_due(store, now=now, retention_days=days), None
            return None, await purge_transcripts(store, memory, now=now, retention_days=days)

    try:
        due, result = asyncio.run(run())
    except RetentionUnavailable as e:
        sys.exit(f"brain purge-transcripts: {e}")

    before = f"ended before {cutoff:%Y-%m-%d %H:%M} UTC ({days}-day retention)"
    if result is None:
        print(f"Dry run, nothing deleted: {len(due)} meetings {before} are due.")
        for meeting in due:
            print(f"  {describe(meeting)}")
        return 0

    print(f"Deleted the transcripts of {len(result.deleted)} meetings {before}.")
    for meeting in result.deleted:
        print(f"  {describe(meeting)}")
    if result.deleted and not result.memory_cleared:
        print("Meeting memory is not configured, so their transcript chunks were not deleted.")
    for meeting_id, error in result.failed.items():
        print(f"  {meeting_id}: failed: {error}")
    return 1 if result.failed else 0


def describe(meeting: Meeting) -> str:
    ended = f"{meeting.ended_at:%Y-%m-%d}" if meeting.ended_at else "?"
    return f"{meeting.id}  ended {ended}  {meeting.title}"
