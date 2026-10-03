"""LiveKit data-channel topics and the payload each one carries.

Public chat uses LiveKit's built-in chat topic. Private questions and answers go over HTTP to
the brain, never through a room broadcast.
"""

from enum import StrEnum

from pydantic import BaseModel

from .agenda import Agenda, AgendaNudge
from .agent import AgentState, FactCheck, ResponseAction, ResponseCard
from .transcript import TranscriptSegment


class Topic(StrEnum):
    TRANSCRIPT = "transcript"  # realtime -> room
    AGENT_STATE = "agent.state"  # realtime -> room
    RESPONSE_CARD = "agent.card"  # realtime -> room
    RESPONSE_ACTION = "agent.card.action"  # board -> realtime
    FACT_CHECK = "agent.fact_check"  # realtime -> room, or one participant when private
    AGENDA = "agent.agenda"  # realtime -> room
    AGENDA_NUDGE = "agent.agenda.nudge"  # realtime -> room
    STAGE = "stage"  # board -> room: snippet id shown on stage, or null


class StagePayload(BaseModel):
    snippet_id: str | None


TOPIC_PAYLOADS: dict[Topic, type[BaseModel]] = {
    Topic.TRANSCRIPT: TranscriptSegment,
    Topic.AGENT_STATE: AgentState,
    Topic.RESPONSE_CARD: ResponseCard,
    Topic.RESPONSE_ACTION: ResponseAction,
    Topic.FACT_CHECK: FactCheck,
    Topic.AGENDA: Agenda,
    Topic.AGENDA_NUDGE: AgendaNudge,
    Topic.STAGE: StagePayload,
}
