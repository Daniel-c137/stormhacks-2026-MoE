from typing import Protocol

from contracts import AgentState
from contracts.agent import AgentStateName

TRANSITIONS: dict[AgentStateName, frozenset[AgentStateName]] = {
    "idle": frozenset({"capturing", "working"}),
    "capturing": frozenset({"working", "idle"}),
    "working": frozenset({"hand_raised", "idle"}),
    "hand_raised": frozenset({"speaking", "idle"}),
    "speaking": frozenset({"followup", "idle"}),
    "followup": frozenset({"capturing", "idle"}),
}


class AgentStateMachine(Protocol):
    """Publishes every change on Topic.AGENT_STATE."""

    @property
    def current(self) -> AgentState: ...

    async def move(self, to: AgentState) -> None: ...
