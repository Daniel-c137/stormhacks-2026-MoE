"""HTTP request and response bodies. Board -> brain, and realtime -> brain (internal)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .agenda import Agenda, AgendaItemStatus, AgendaNudge
from .agent import AgentState, Answer, CodeSnippet, FactCheck, Invocation, Visibility
from .meeting import Meeting, Person
from .transcript import TranscriptSegment

AskTurnRole = Literal["user", "agent"]


class LoginRequest(BaseModel):
    """POST /auth/login. There is no public sign-up; accounts come from `brain add-user`."""

    email: str
    password: str


class LoginResponse(BaseModel):
    """The board sends `token` as `Authorization: Bearer <token>` until `expires_at`."""

    token: str
    expires_at: datetime
    person: Person


class PasswordChange(BaseModel):
    """POST /auth/password while signed in. The new password is at least 10 characters."""

    current_password: str
    new_password: str


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


class CatchUpRequest(BaseModel):
    """realtime -> brain when someone joins a live meeting 5 minutes or more after it started,
    or comes back after 5 minutes or more away. `since` and `until` are seconds from the meeting
    start: the span they missed."""

    participant_id: str
    since: float = Field(ge=0, allow_inf_nan=False)
    until: float = Field(ge=0, allow_inf_nan=False)


class CatchUpResponse(BaseModel):
    """What the worker sends only to the participant, as a private chat message from the agent
    (Topic.PRIVATE_CHAT, recipient_id set). `text` is None when there is nothing to send.
    `source_times` are the meeting seconds of the transcript lines it rests on. Never stored,
    broadcast or spoken."""

    text: str | None = None
    source_times: list[float] = []


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


class KeytermsResponse(BaseModel):
    """Words the worker's Scribe streams are biased toward, most important first, already within
    Scribe Realtime's limits (brain.keyterms)."""

    terms: list[str]


class WorkerMeetingResponse(BaseModel):
    """realtime -> brain when the worker joins a meeting's room (the room is named by the meeting
    id) and before it speaks. Segment times are seconds from meeting.started_at. voice_id is the
    team's chosen agent voice; None means the worker's default (ELEVENLABS_VOICE_ID)."""

    meeting: Meeting
    voice_id: str | None = None
