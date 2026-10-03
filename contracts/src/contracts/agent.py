from datetime import datetime
from typing import Literal

from pydantic import BaseModel

AgentStateName = Literal["idle", "capturing", "working", "hand_raised", "speaking", "followup"]
HandUrgency = Literal["normal", "critical"]
Visibility = Literal["public", "private"]
InvocationVia = Literal["voice", "follow_up", "chat", "ask", "private"]
SourceKind = Literal[
    "meeting", "github_issue", "github_pr", "github_code", "github_release", "jira_issue"
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
    """Evidence behind an answer: a meeting moment, a GitHub item or a Jira key."""

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
    github_url: str
    caption: str
    highlight: tuple[int, int] | None = None  # UI-only


class Answer(BaseModel):
    id: str
    invocation_id: str
    text: str
    sources: list[Source] = []
    snippets: list[CodeSnippet] = []
    unavailable: list[str] = []  # sources that were missing or failed; never papered over


ResponseCardStatus = Literal["pending", "spoken", "sent_to_chat", "dismissed"]
ResponseActionName = Literal["speak", "send_to_chat", "dismiss", "show_on_stage"]


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
    id: str
    claim: str
    speaker_name: str
    verdict: Verdict
    confidence: float
    severity: Severity
    snippet_ids: list[str] = []
    sources: list[Source] = []
    raised_hand: bool = False
    visibility: Visibility = "public"
    recipient_id: str | None = None  # set when visibility is private
    t: float | None = None
    created_at: datetime | None = None
