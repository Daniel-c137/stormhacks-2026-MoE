"""Polaris's state as the room sees it: every change is published on Topic.AGENT_STATE, and only
the moves in TRANSITIONS are made."""

import pytest

from contracts import AgentState, Topic
from realtime_worker.state import TRANSITIONS, IllegalMove, RoomAgentState

pytestmark = pytest.mark.anyio


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, topic, payload, *, to=None):
        self.published.append((topic, payload, to))


def state(name: str, detail: str = "") -> AgentState:
    return AgentState(state=name, detail=detail)


async def test_starts_idle_and_publishes_every_change_to_everyone():
    bus = Bus()
    machine = RoomAgentState(bus)
    assert machine.current.state == "idle"

    await machine.move(state("working", "Looking into the refund window"))

    assert machine.current.state == "working"
    assert bus.published == [
        (Topic.AGENT_STATE, state("working", "Looking into the refund window"), None)
    ]


async def test_an_illegal_move_is_refused_and_nothing_is_published():
    bus = Bus()
    machine = RoomAgentState(bus)

    assert not machine.can_move("followup")
    with pytest.raises(IllegalMove):
        await machine.move(state("followup"))

    assert machine.current.state == "idle"
    assert bus.published == []


async def test_the_same_state_again_is_published_only_when_it_changed():
    bus = Bus()
    machine = RoomAgentState(bus)

    await machine.move(state("idle", ""))
    await machine.move(state("idle", "Couldn't answer: the brain is unavailable"))

    assert [p.detail for _, p, _ in bus.published] == ["Couldn't answer: the brain is unavailable"]


def test_the_moves_the_meeting_needs_are_allowed():
    needed = [
        ("idle", "capturing"),  # the Ask button
        ("capturing", "working"),  # the question came
        ("capturing", "idle"),  # the Ask was cancelled or ran out
        ("capturing", "hand_raised"),  # ... while an answer was waiting
        ("idle", "working"),  # a voice question or a chat mention
        ("working", "hand_raised"),  # the answer card is ready
        ("working", "idle"),  # the answer failed, or went to chat
        ("hand_raised", "speaking"),  # someone chose Speak
        ("hand_raised", "working"),  # a new question while an answer waits
        ("hand_raised", "capturing"),  # an Ask while an answer waits
        ("hand_raised", "idle"),  # the card was posted or dismissed
        ("idle", "hand_raised"),  # an answer card is waiting
        ("idle", "speaking"),  # Speak on a card after the hand went down
        ("speaking", "idle"),
        ("speaking", "hand_raised"),  # another answer is still waiting
    ]
    for before, after in needed:
        assert after in TRANSITIONS[before], (before, after)
