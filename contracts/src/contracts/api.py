"""HTTP request and response bodies. Board -> brain, and realtime -> brain (internal)."""

from pydantic import BaseModel

from .agent import Answer, Invocation, Visibility
from .meeting import Meeting
from .transcript import TranscriptSegment


class CreateMeetingRequest(BaseModel):
    title: str


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
