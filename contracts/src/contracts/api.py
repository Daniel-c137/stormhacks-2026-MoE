"""HTTP request and response bodies. Board -> brain, and realtime -> brain (internal)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .agenda import Agenda, AgendaItemStatus, AgendaNudge
from .agent import Answer, CodeSnippet, FactCheck, Invocation, Visibility
from .meeting import Meeting, Person
from .transcript import TranscriptSegment

AskTurnRole = Literal["user", "agent"]


class LoginRequest(BaseModel):
    """POST /auth/login. Accounts come from `brain add-user`, an admin (POST /team/accounts), or
    an invited email signing up (SignupRequest)."""

    email: str
    password: str


class SignupRequest(BaseModel):
    """POST /auth/signup (#128): only an email an admin invited (a person on a team with that
    email and no login yet) creates an account, on that team. Answered with a LoginResponse:
    signed in at once."""

    name: str
    email: str
    password: str


class AuthOptions(BaseModel):
    """GET /auth/options: what the sign-in page can offer. signup: the brain signs sessions
    (AUTH_SECRET); sign-up is always invite-only. google: the brain has a Google OAuth client
    and its public callback (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URL)."""

    signup: bool
    google: bool


class GoogleExchangeRequest(BaseModel):
    """POST /auth/google/exchange: the one-time code the brain's Google callback sent the board,
    swapped for a session (a LoginResponse). The session token is never put in a URL, and the code
    only works with the HttpOnly google_handoff cookie the callback set in the same browser."""

    code: str


class LoginResponse(BaseModel):
    """The board sends `token` as `Authorization: Bearer <token>` until `expires_at`."""

    token: str
    expires_at: datetime
    person: Person


class PasswordChange(BaseModel):
    """POST /auth/password while signed in. The new password is at least 10 characters."""

    current_password: str
    new_password: str


class CreateAccountRequest(BaseModel):
    """POST /team/accounts, admin only: a person on the admin's own team with an email login.
    With invite, the person gets no login: they create it themselves on the sign-in page, with
    a password or Google (SignupRequest)."""

    name: str
    email: str
    title: str | None = None
    is_admin: bool = False
    invite: bool = False


class CreateAccountResponse(BaseModel):
    """The generated password is returned only this once; nothing is emailed. None for an
    invite."""

    person: Person
    password: str | None


class CreateMeetingRequest(BaseModel):
    """Without scheduled_start the meeting starts now; with it, the meeting waits as scheduled."""

    title: str
    scheduled_start: datetime | None = None
    duration_min: int | None = None
    invitee_ids: list[str] = []
    translate: bool = False  # live translation of non-English speech (#106)


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
    """The worker sends each check to its recipient_id only, as a private chat message from the
    agent (Topic.PRIVATE_CHAT), never stored, spoken or shown to the room. `snippets` are the code
    the checks' snippet_ids name."""

    checks: list[FactCheck] = []
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


class TranslationUpdate(BaseModel):
    """board -> brain: the host switches live translation on or off before anyone joins."""

    translate: bool


class TranslateRequest(BaseModel):
    """realtime -> brain: speech to put into English. `language` is Scribe's detected ISO 639-1
    code when it gave one; None asks the brain to detect it."""

    text: str
    language: str | None = None


class TranslateResponse(BaseModel):
    """`language` is the detected (or given) language; `text` is English, unchanged when the
    speech already was."""

    language: str
    text: str


class InvokeRequest(BaseModel):
    invocation: Invocation
    recent_segments: list[TranscriptSegment] = []


class InvokeResponse(BaseModel):
    answer: Answer


class KeytermsResponse(BaseModel):
    """Words the worker's Scribe streams are biased toward, most important first, already within
    Scribe Realtime's limits (brain.keyterms)."""

    terms: list[str]


class CardPermissionResponse(BaseModel):
    """realtime -> brain before the worker acts on a shared answer card (Speak, Post in chat,
    Dismiss) for a participant: whether TeamSettings.who_can_allow lets them. With "host", only
    the meeting's host or an admin; with "everyone", any participant. Never the agent itself."""

    allowed: bool


class WorkerMeetingResponse(BaseModel):
    """realtime -> brain when the worker joins a meeting's room (the room is named by the meeting
    id) and before it speaks. Segment times are seconds from meeting.started_at. voice_id is the
    team's chosen agent voice; None means the worker's default (ELEVENLABS_VOICE_ID)."""

    meeting: Meeting
    voice_id: str | None = None
