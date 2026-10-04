"""LiveKit data-channel topics and the payload each one carries.

Public chat uses LiveKit's built-in chat topic. Private questions and answers with the agent go
over HTTP to the brain, never through a room broadcast. A private message, between two people or
a fact-check from the agent to whoever made the claim, is sent to that one participant only and is
never stored.
"""

from enum import StrEnum

from pydantic import BaseModel

from .agenda import Agenda, AgendaNudge
from .agent import AgentState, ResponseAction, ResponseCard
from .chat import ChatMessage
from .transcript import TranscriptSegment


class Topic(StrEnum):
    TRANSCRIPT = "transcript"  # realtime -> room
    AGENT_STATE = "agent.state"  # realtime -> room
    # board -> realtime: the Ask button; the presser's next final segment is the question
    ASK = "agent.ask"
    RESPONSE_CARD = "agent.card"  # realtime -> room
    RESPONSE_ACTION = "agent.card.action"  # board -> realtime
    AGENDA = "agent.agenda"  # realtime -> room
    AGENDA_NUDGE = "agent.agenda.nudge"  # realtime -> room
    STAGE = "stage"  # board -> room: snippet id shown on stage, or null
    PRIVATE_CHAT = "chat.private"  # board or realtime -> one participant; ephemeral


class StagePayload(BaseModel):
    snippet_id: str | None


class AskSignal(BaseModel):
    by_id: str
    cancel: bool = False  # True withdraws a pending Ask press


TOPIC_PAYLOADS: dict[Topic, type[BaseModel]] = {
    Topic.TRANSCRIPT: TranscriptSegment,
    Topic.AGENT_STATE: AgentState,
    Topic.ASK: AskSignal,
    Topic.RESPONSE_CARD: ResponseCard,
    Topic.RESPONSE_ACTION: ResponseAction,
    Topic.AGENDA: Agenda,
    Topic.AGENDA_NUDGE: AgendaNudge,
    Topic.STAGE: StagePayload,
    Topic.PRIVATE_CHAT: ChatMessage,
}
