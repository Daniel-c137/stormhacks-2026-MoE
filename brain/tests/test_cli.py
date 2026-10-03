from pathlib import Path

import pytest

from brain.cli import main
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


def test_report_without_gemini_config_says_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")

    with pytest.raises(SystemExit) as exit_info:
        main(["report", str(FIXTURES / "standup.json"), "--out", str(tmp_path / "r.json")])

    assert "GEMINI_API_KEY" in str(exit_info.value.code)
    assert not (tmp_path / "r.json").exists()
