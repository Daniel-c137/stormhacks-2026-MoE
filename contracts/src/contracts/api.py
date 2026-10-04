"""HTTP request and response bodies. Board -> brain, and realtime -> brain (internal)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .agenda import Agenda, AgendaItemStatus, AgendaNudge
from .agent import AgentState, Answer, CodeSnippet, FactCheck, Invocation, Visibility
from .meeting import Meeting
from .transcript import TranscriptSegment

AskTurnRole = Literal["user", "agent"]


class CreateMeetingRequest(BaseModel):
    """Without scheduled_start the meeting starts now; with it, the meeting waits as scheduled."""

    title: str
    scheduled_start: datetime | None = None
    duration_min: int | None = None
    invitee_ids: list[str] = []


class JoinMeetingResponse(BaseModel):
    meeting: Meeting
    livekit_url: str
    token: str


class InviteRequest(BaseModel):
    person_ids: list[str]


class AskTurn(BaseModel):
    """An earlier turn of the same conversation, sent back so a follow-up has its context."""

    role: AskTurnRole
    text: str


class AskRequest(BaseModel):
    """A typed question from the board. Private answers go back only to the asker."""

    question: str
    visibility: Visibility
    history: list[AskTurn] = []  # oldest first


class AgendaItemInput(BaseModel):
    """An agenda item as a person edits it. No id means a new item."""

    id: str | None = None
    title: str
    minutes: int | None = None
    status: AgendaItemStatus | None = None  # None keeps the item's status; new items are pending


class AgendaUpdate(BaseModel):
    """The whole edited list, in order. Items left out are removed."""

    items: list[AgendaItemInput]


class AgendaRewriteRequest(BaseModel):
    text: str


class AgendaRewriteResponse(BaseModel):
    text: str


class AgendaTrackRequest(BaseModel):
    """realtime -> brain on a timer. `now` is seconds from the meeting start; omitted, the
    brain takes it from started_at. A finite time no later than the real time since the start
    (plus a minute of slack)."""

    now: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class AgendaTrackResponse(BaseModel):
    """The worker publishes `agenda` on Topic.AGENDA and each nudge on Topic.AGENDA_NUDGE."""

    agenda: Agenda
    nudges: list[AgendaNudge] = []


class FactCheckRequest(BaseModel):
    """realtime -> brain on a timer, never per utterance. `now` is seconds from the meeting
    start; omitted, the brain takes it from started_at."""

    now: float | None = Field(default=None, ge=0)


class FactCheckResponse(BaseModel):
    """The worker publishes each check on Topic.FACT_CHECK: a public one to the room, a private
    one only to its recipient_id. `agent_state`, set only when a check raises the hand, goes on
    Topic.AGENT_STATE; the hand is a visual cue and nothing is spoken. `snippets` are the code
    the checks' snippet_ids name, for whoever sees those checks."""

    checks: list[FactCheck] = []
    agent_state: AgentState | None = None
    snippets: list[CodeSnippet] = []


class ProfileUpdate(BaseModel):
    """Only the fields that are set change."""

    name: str | None = None


class SegmentsIngest(BaseModel):
    segments: list[TranscriptSegment]


class InvokeRequest(BaseModel):
    invocation: Invocation
    recent_segments: list[TranscriptSegment] = []


class InvokeResponse(BaseModel):
    answer: Answer
