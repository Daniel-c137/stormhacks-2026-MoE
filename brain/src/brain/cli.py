"""`brain report` turns a transcript into a review file of the report and task drafts.
`brain push` creates Jira issues for the drafts a named person approved.
`brain migrate` applies supabase/migrations to DATABASE_URL.
`brain purge-transcripts` deletes transcripts older than TRANSCRIPT_RETENTION_DAYS."""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from brain.config import Settings
from brain.db import migrate
from brain.jira import ApprovalRequired, JiraPusher, JiraUnavailable, apply_results, jira_config
from brain.llm import LLM, LLMError, MockLLM, make_llm
from brain.report import ProcessedMeeting, ReportExtraction, build_report, load_transcript
from brain.retention import (
    RetentionUnavailable,
    open_retention_stores,
    purge_transcripts,
    retention_cutoff,
    transcripts_due,
)
from contracts import Meeting, TaskPushRequest


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

    migrate_cmd = commands.add_parser("migrate", help="apply supabase/migrations to DATABASE_URL")
    migrate_cmd.set_defaults(run=run_migrate)

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
    source = source or f"Gemini {llm.last_model}"

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
