"""`brain report`: a transcript in, a review file of the report and task drafts out."""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from brain.llm import LLM, LLMError, MockLLM, make_llm
from brain.report import ProcessedMeeting, ReportExtraction, build_report, load_transcript


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

    args = parser.parse_args(argv)
    return args.run(args)


def run_report(args: argparse.Namespace) -> int:
    meeting = load_transcript(args.transcript, started_at=args.date)
    llm: LLM
    if args.mock_response:
        scripted = ReportExtraction.model_validate_json(args.mock_response.read_text())
        llm = MockLLM(structured={ReportExtraction: scripted})
        source = "MockLLM (scripted, not Gemini)"
    else:
        try:
            llm = make_llm()
        except LLMError as e:
            sys.exit(f"brain report: {e}")
        source = "Gemini"

    try:
        report = asyncio.run(build_report(llm, meeting))
    except (LLMError, ValueError) as e:
        sys.exit(f"brain report: {e}")

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
