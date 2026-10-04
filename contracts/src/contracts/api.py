"""HTTP request and response bodies. Board -> brain, and realtime -> brain (internal)."""

from datetime import datetime

from pydantic import BaseModel

from .agenda import AgendaItem
from .agent import Answer, Invocation, Visibility
from .meeting import Meeting
from .transcript import TranscriptSegment


class CreateMeetingRequest(BaseModel):
    """No scheduled_for starts the meeting now. Invitees are team members; the creator organises."""

    title: str
    scheduled_for: datetime | None = None
    duration_min: int | None = None
    invitee_ids: list[str] = []


class UpdateProfileRequest(BaseModel):
    """The signed-in user's own profile. photo is an image data URL; None removes it."""

    name: str
    photo: str | None = None


class UpdateAgendaRequest(BaseModel):
    """Topics people add in the lobby replace the meeting's agenda items."""

    items: list[AgendaItem]


class RewriteTopicRequest(BaseModel):
    """Tidy one agenda topic. The text comes back for the user to accept; nothing is saved."""

    text: str


class RewriteTopicResponse(BaseModel):
    text: str


class JoinMeetingResponse(BaseModel):
    meeting: Meeting
    livekit_url: str
    token: str


class AskRequest(BaseModel):
    """A typed question from the board. Private answers go back only to the asker."""

    question: str
    visibility: Visibility


class SegmentsIngest(BaseModel):
    segments: list[TranscriptSegment]


class InvokeRequest(BaseModel):
    invocation: Invocation
    recent_segments: list[TranscriptSegment] = []


class InvokeResponse(BaseModel):
    answer: Answer
