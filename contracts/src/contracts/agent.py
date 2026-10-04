from datetime import datetime
from typing import Literal

from pydantic import BaseModel

AgentStateName = Literal["idle", "capturing", "working", "hand_raised", "speaking", "followup"]
HandUrgency = Literal["normal", "critical"]
Visibility = Literal["public", "private"]
InvocationVia = Literal["voice", "follow_up", "chat", "ask", "private"]
SourceKind = Literal[
    "meeting",
    "github_issue",
    "github_pr",
    "github_code",
    "github_release",
    "gitlab_issue",
    "gitlab_mr",
    "gitlab_code",
    "gitlab_release",
    "jira_issue",
]
Verdict = Literal["supported", "contradicted", "unknown"]
Severity = Literal["low", "high"]


class AgentState(BaseModel):
    state: AgentStateName
    detail: str
    hand_urgency: HandUrgency = "normal"
    hand_reason: str = ""


class Invocation(BaseModel):
    """A deliberate request to the agent. The agent never reasons on every utterance."""

    id: str
    meeting_id: str
    via: InvocationVia
    visibility: Visibility
    asked_by_id: str
    asked_by_name: str
    question: str
    t: float | None = None


class Source(BaseModel):
    """Evidence behind an answer: a meeting moment, a GitHub or GitLab item or a Jira key. A
    repository's items name it: dropsubs/web#41, group/project!12 (a GitLab merge request)."""

    kind: SourceKind
    label: str
    url: str | None = None
    meeting_id: str | None = None
    t: float | None = None


class CodeSnippet(BaseModel):
    id: str
    path: str
    start_line: int
    end_line: int
    language: str
    code: str
    github_url: str  # the lines on their code host: GitHub, or GitLab for a GitLab project
    caption: str
    repo: str | None = None  # owner/name on GitHub, the project path on GitLab
    highlight: tuple[int, int] | None = None  # UI-only


class Answer(BaseModel):
    id: str
    invocation_id: str
    text: str
    sources: list[Source] = []
    snippets: list[CodeSnippet] = []
    unavailable: list[str] = []  # sources that were missing or failed; never papered over


ResponseCardStatus = Literal["pending", "speaking", "spoken", "sent_to_chat", "dismissed"]
# stop: cut the answer being spoken off; the card goes back to pending.
ResponseActionName = Literal["speak", "stop", "send_to_chat", "dismiss", "show_on_stage"]


class ResponseCard(BaseModel):
    """Shared answer to a voice question. Silent until a participant chooses Speak."""

    id: str
    meeting_id: str
    invocation: Invocation
    answer: Answer
    status: ResponseCardStatus = "pending"
    urgency: HandUrgency = "normal"


class ResponseAction(BaseModel):
    card_id: str
    action: ResponseActionName
    by_id: str


class QuestionAnswered(BaseModel):
    id: str
    asked_by: str
    question: str
    answer: str
    snippet_id: str | None = None
    via: InvocationVia
    t: float | None = None


class FactCheck(BaseModel):
    """A claim checked against the team's records during a live meeting. It is never shown to the
    room: the agent sends it as a private chat message to recipient_id, the participant who made
    the claim. The copy kept for the write-up has no recipient_id."""

    id: str
    claim: str
    speaker_name: str
    verdict: Verdict
    confidence: float
    severity: Severity
    finding: str = ""  # what the records show, in one short sentence
    snippet_ids: list[str] = []
    sources: list[Source] = []
    recipient_id: str | None = None  # the claimant's participant id
    t: float | None = None
    created_at: datetime | None = None
