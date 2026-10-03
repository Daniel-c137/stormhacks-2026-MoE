"""End to end over real HTTP: `brain report` (MockLLM) -> review file -> `brain push` -> a Jira
MCP server on localhost. No external services."""

from pathlib import Path

from brain.cli import main
from brain.report import ProcessedMeeting

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_transcript_to_reviewed_drafts_to_jira_issues(jira_env, tmp_path, capsys):
    review_file = tmp_path / "standup.review.json"

    assert (
        main(
            [
                "report",
                str(FIXTURES / "standup.json"),
                "--mock-response",
                str(FIXTURES / "standup.extraction.json"),
                "--out",
                str(review_file),
            ]
        )
        == 0
    )
    assert jira_env.created == []  # a report alone never writes to Jira

    assert main(["push", str(review_file), "--all", "--approved-by", "Alice Moreau"]) == 0

    assert [c["summary"] for c in jira_env.created] == [
        "Refund the 14 double-charged users",
        "Find an owner for the model retirement",
    ]
    assert jira_env.created[0]["additional_fields"] == {"duedate": "2026-10-07"}
    tasks = ProcessedMeeting.model_validate_json(review_file.read_text()).report.tasks
    assert [(t.key, t.jira_status) for t in tasks] == [("DS-117", "todo"), ("DS-118", "todo")]
    assert "https://dropsubs.atlassian.net/browse/DS-117" in capsys.readouterr().out

    # Pushing the same review again creates nothing new.
    assert main(["push", str(review_file), "--all", "--approved-by", "Alice Moreau"]) == 0
    assert len(jira_env.created) == 2
