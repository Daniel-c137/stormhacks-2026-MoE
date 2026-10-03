"""Shared contract. contracts/ts mirrors these modules; change both in the same commit."""

from .agenda import Agenda, AgendaItem, AgendaNudge
from .agent import (
    AgentState,
    Answer,
    CodeSnippet,
    FactCheck,
    Invocation,
    QuestionAnswered,
    ResponseAction,
    ResponseCard,
    Source,
)
from .api import (
    AskRequest,
    CreateMeetingRequest,
    InvokeRequest,
    InvokeResponse,
    JoinMeetingResponse,
    SegmentsIngest,
)
from .chat import ChatMessage
from .events import TOPIC_PAYLOADS, StagePayload, Topic
from .identity import AGENT_PARTICIPANT_ID, Identity, get_identity
from .meeting import Meeting, Participant, Person, Team
from .report import (
    Decision,
    DecisionRelation,
    Report,
    ReportProgress,
    Risk,
    TaskDraft,
    TaskPushRequest,
    TaskPushResult,
)
from .settings import GitHubSettings, JiraSettings, TeamSettings, Voice
from .transcript import TranscriptSegment

__all__ = [
    "AGENT_PARTICIPANT_ID",
    "TOPIC_PAYLOADS",
    "Agenda",
    "AgendaItem",
    "AgendaNudge",
    "AgentState",
    "Answer",
    "AskRequest",
    "ChatMessage",
    "CodeSnippet",
    "CreateMeetingRequest",
    "Decision",
    "DecisionRelation",
    "FactCheck",
    "GitHubSettings",
    "Identity",
    "Invocation",
    "InvokeRequest",
    "InvokeResponse",
    "JiraSettings",
    "JoinMeetingResponse",
    "Meeting",
    "Participant",
    "Person",
    "QuestionAnswered",
    "Report",
    "ReportProgress",
    "ResponseAction",
    "ResponseCard",
    "Risk",
    "SegmentsIngest",
    "Source",
    "StagePayload",
    "TaskDraft",
    "TaskPushRequest",
    "TaskPushResult",
    "Team",
    "TeamSettings",
    "Topic",
    "TranscriptSegment",
    "Voice",
    "get_identity",
]
