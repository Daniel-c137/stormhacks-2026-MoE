from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from brain.cli import main
from brain.llm import GeminiLLM
from brain.report import ProcessedMeeting

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


def test_report_names_the_gemini_model_that_answered(monkeypatch, tmp_path, capsys):
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
    assert "Gemini spare-model" in capsys.readouterr().out


def test_report_without_gemini_config_says_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")

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
