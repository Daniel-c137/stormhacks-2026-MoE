from typing import Protocol

from contracts import AgentState, Topic
from contracts.agent import AgentStateName

from .room import RoomBus

# capturing is listening for a question (the Ask button); hand_raised is an answer card waiting,
# or a fact-check's raised hand.
TRANSITIONS: dict[AgentStateName, frozenset[AgentStateName]] = {
    "idle": frozenset({"capturing", "working", "hand_raised", "speaking"}),
    "capturing": frozenset({"working", "idle", "hand_raised"}),
    "working": frozenset({"hand_raised", "idle"}),
    # a new question or an Ask press while an answer is waiting
    "hand_raised": frozenset({"speaking", "idle", "working", "capturing"}),
    # after speaking, back to idle, or to another answer still waiting
    "speaking": frozenset({"followup", "idle", "hand_raised"}),
    "followup": frozenset({"capturing", "idle"}),
}


class IllegalMove(ValueError):
    """The move is not in TRANSITIONS; nothing was published."""


class AgentStateMachine(Protocol):
    """Publishes every change on Topic.AGENT_STATE."""

    @property
    def current(self) -> AgentState: ...

    def can_move(self, to: AgentStateName) -> bool: ...

    async def move(self, to: AgentState) -> None: ...


class RoomAgentState:
    """The agent's state for the whole room. A move to the same state is published only when
    something about it changed (its detail, say)."""

    def __init__(self, bus: RoomBus):
        self._bus = bus
        self._current = AgentState(state="idle", detail="")

    @property
    def current(self) -> AgentState:
        return self._current

    def can_move(self, to: AgentStateName) -> bool:
        return to == self._current.state or to in TRANSITIONS[self._current.state]

    async def move(self, to: AgentState) -> None:
        if not self.can_move(to.state):
            raise IllegalMove(f"{self._current.state} -> {to.state}")
        if to == self._current:
            return
        self._current = to
        await self._bus.publish(Topic.AGENT_STATE, to)
