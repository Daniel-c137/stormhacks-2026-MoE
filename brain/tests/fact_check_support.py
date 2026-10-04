"""Scripting the fact-checker's two model calls, and a GitHub where PR #41 was merged after the
latest release, so "PR 41 is released" is contradicted."""

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from api_support import ALEX, TEAM
from ask_support import ids_for

from brain.agent.ask import PlannedCall
from brain.agent.factcheck import ClaimPlan, ClaimVerdict, FactCheckPlan, FactCheckVerdicts
from brain.llm import MockLLM
from contracts import (
    GitHubSettings,
    JiraSettings,
    Meeting,
    Person,
    TeamSettings,
    TranscriptSegment,
)

CONNECTED = {
    "jira_mcp_url": "http://unused.invalid/mcp",
    "jira_base_url": "https://dropsubs.atlassian.net",
    "github_mcp_url": "http://unused.invalid/github",
}

RELEASED = "The refund fix from PR 41 is already released."
CLOSED = "DS-104 was closed yesterday."
SHIPPED = "We shipped v0.9.3 on Monday."

PR_41: dict[str, Any] = {
    "number": 41,
    "title": "Fix the refund double charge",
    "state": "closed",
    "merged": True,
    "merged_at": "2026-09-30T15:00:00Z",
    "html_url": "https://github.com/dropsubs/app/pull/41",
    "body": "Refunds are charged once.",
}
RELEASES: list[dict[str, Any]] = [
    {
        "tag_name": "v0.9.3",
        "name": "v0.9.3",
        "body": "Signup fixes.",
        "target_commitish": "main",
        "published_at": "2026-09-28T12:00:00Z",
        "html_url": "https://github.com/dropsubs/app/releases/tag/v0.9.3",
    },
    {
        "tag_name": "v0.9.2",
        "name": "v0.9.2",
        "body": "Billing page.",
        "target_commitish": "main",
        "published_at": "2026-09-14T12:00:00Z",
        "html_url": "https://github.com/dropsubs/app/releases/tag/v0.9.2",
    },
]

FINDING = "PR #41 was merged on 30 September, after the latest release, v0.9.3, on 28 September."

READ_41 = PlannedCall(tool="github_read", number=41, kind="pr")
RELEASES_CALL = PlannedCall(tool="github_releases")


def merged_after_release(fake_github) -> None:
    """PR #41 merged on 30 September; the latest release, v0.9.3, is from 28 September."""
    fake_github.issues.pop(41, None)
    fake_github.pulls[41] = dict(PR_41)
    fake_github.releases = [dict(r) for r in RELEASES]


def check(claim: str = "c1", *calls: PlannedCall, code_query: str | None = None) -> ClaimPlan:
    return ClaimPlan(claim=claim, calls=list(calls), code_query=code_query)


def plan(*claims: ClaimPlan) -> FactCheckPlan:
    return FactCheckPlan(claims=list(claims))


Verdict = Callable[[str], ClaimVerdict]


def verdict(
    claim: str = "c1",
    verdict: str = "contradicted",
    *,
    confidence: float = 0.9,
    severity: str = "high",
    cite: tuple[str, ...] = ("dropsubs/app#41", "dropsubs/app@v0.9.3"),
    extra_ids: tuple[str, ...] = (),
    finding: str = FINDING,
) -> Verdict:
    """A verdict citing the evidence lines that contain each of `cite`, plus `extra_ids` as is."""

    def answer(prompt: str) -> ClaimVerdict:
        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            severity=severity,
            evidence_ids=[*ids_for(prompt, *cite), *extra_ids],
            finding=finding,
        )

    return answer


def scripted(claims: FactCheckPlan, *verdicts: Verdict) -> MockLLM:
    return MockLLM(
        structured={
            FactCheckPlan: claims,
            FactCheckVerdicts: lambda prompt: FactCheckVerdicts(
                checks=[v(prompt) for v in verdicts]
            ),
        }
    )


CONTRADICTED = (plan(check("c1", READ_41, RELEASES_CALL)), verdict())


async def live(store, **team) -> Meeting:
    """A live meeting of TEAM, whose settings name the repo and Jira project."""
    await store.save_settings(
        TeamSettings(
            team_id=TEAM.id,
            github=GitHubSettings(repo="dropsubs/app"),
            jira=JiraSettings(project="DS"),
            **team,
        )
    )
    return await store.create_meeting(TEAM.id, "Refund sync", ALEX.id)


async def say(store, meeting: Meeting, *lines: tuple[Person, str, float]) -> None:
    """Final segments, each five seconds long from the given start."""
    segments = [
        TranscriptSegment(
            seg_id=uuid4().hex,
            meeting_id=meeting.id,
            speaker_id=speaker.id,
            speaker_name=speaker.name,
            text=text,
            is_final=True,
            t_start=start,
            t_end=start + 5,
        )
        for speaker, text, start in lines
    ]
    await store.add_segments(meeting.id, segments)
