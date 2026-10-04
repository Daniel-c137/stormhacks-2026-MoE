"""The room bus: typed payloads over LiveKit data packets. What participants may send the worker is
a short list, and who sent it is what LiveKit says, never what the payload claims."""

import asyncio
import json

import pytest

from contracts import (
    AGENT_PARTICIPANT_ID,
    TOPIC_PAYLOADS,
    AgentState,
    AskSignal,
    ResponseAction,
    StagePayload,
    Topic,
)
from realtime_worker.bus import BOARD_TOPICS, LiveKitBus

pytestmark = pytest.mark.anyio


class FakeLocalParticipant:
    def __init__(self):
        self.sent: list[dict] = []

    async def publish_data(self, payload, *, reliable=True, destination_identities=(), topic=""):
        self.sent.append(
            {
                "payload": payload,
                "reliable": reliable,
                "to": list(destination_identities),
                "topic": topic,
            }
        )


def packet(payload) -> bytes:
    return json.dumps(payload).encode()


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.fixture
def participant():
    return FakeLocalParticipant()


@pytest.fixture
async def bus(participant):
    bus = LiveKitBus(participant)
    yield bus
    await bus.aclose()


@pytest.fixture
def received(bus):
    got: list[tuple[Topic, object, str]] = []

    def on(topic: Topic):
        async def handler(payload, sender: str) -> None:
            got.append((topic, payload, sender))

        bus.subscribe(topic, handler)

    for topic in Topic:
        on(topic)
    return got


def test_ask_is_a_board_topic_carrying_an_ask_signal():
    assert Topic.ASK == "agent.ask"
    assert TOPIC_PAYLOADS[Topic.ASK] is AskSignal
    assert AskSignal(by_id="u-alex").cancel is False


# publishing


async def test_publishes_json_reliably_on_the_topic_to_everyone(bus, participant):
    await bus.publish(Topic.AGENT_STATE, AgentState(state="working", detail="Looking it up"))

    [sent] = participant.sent
    assert sent["topic"] == "agent.state"
    assert sent["reliable"] is True
    assert sent["to"] == []
    assert json.loads(sent["payload"]) == {
        "state": "working",
        "detail": "Looking it up",
        "hand_urgency": "normal",
        "hand_reason": "",
    }


async def test_publishes_to_one_participant_only_when_asked(bus, participant):
    await bus.publish(Topic.AGENT_STATE, AgentState(state="idle", detail=""), to=["u-alex"])

    assert participant.sent[0]["to"] == ["u-alex"]


async def test_an_empty_recipient_list_is_refused_rather_than_broadcast(bus, participant):
    """LiveKit sends to everyone when the list is empty: a private payload would leak."""
    with pytest.raises(ValueError):
        await bus.publish(Topic.AGENT_STATE, AgentState(state="idle", detail=""), to=[])

    assert participant.sent == []


async def test_a_payload_of_the_wrong_type_for_the_topic_is_refused(bus, participant):
    with pytest.raises(TypeError):
        await bus.publish(Topic.AGENDA, AgentState(state="idle", detail=""))

    assert participant.sent == []


# receiving


def test_participants_may_only_send_the_board_topics():
    assert BOARD_TOPICS == {Topic.RESPONSE_ACTION, Topic.ASK, Topic.STAGE}


async def test_a_board_topic_reaches_its_handler_with_the_livekit_sender(bus, received):
    bus.receive(packet({"by_id": "u-alex", "cancel": False}), "u-alex", "agent.ask")
    await settle()

    assert received == [(Topic.ASK, AskSignal(by_id="u-alex"), "u-alex")]


async def test_the_sender_comes_from_livekit_not_from_the_payload(bus, received):
    action = {"card_id": "c-1", "action": "speak", "by_id": "u-sarah"}

    bus.receive(packet(action), "u-alex", "agent.card.action")
    await settle()

    [(_, payload, sender)] = received
    assert isinstance(payload, ResponseAction)
    assert sender == "u-alex"  # the handler decides what a mismatched by_id means


@pytest.mark.parametrize(
    "topic",
    ["transcript", "agent.state", "agent.card", "agent.fact_check", "agent.agenda"],
)
async def test_agent_topics_from_a_participant_are_ignored(bus, received, topic):
    """Only the agent publishes these; a participant sending one is forging it."""
    bus.receive(packet({"state": "speaking", "detail": "forged"}), "u-alex", topic)
    await settle()

    assert received == []


async def test_unknown_topics_and_chat_are_ignored(bus, received):
    bus.receive(packet({"message": "hi"}), "u-alex", "lk-chat-topic")
    bus.receive(packet({"x": 1}), "u-alex", "chat.private")
    bus.receive(packet({"x": 1}), "u-alex", None)
    await settle()

    assert received == []


async def test_a_payload_that_does_not_validate_is_ignored(bus, received):
    bus.receive(b"not json", "u-alex", "agent.ask")
    bus.receive(packet({"cancel": True}), "u-alex", "agent.ask")  # no by_id
    bus.receive(
        packet({"card_id": "c-1", "action": "shout", "by_id": "u-alex"}),
        "u-alex",
        "agent.card.action",
    )
    await settle()

    assert received == []


async def test_packets_without_a_participant_sender_are_ignored(bus, received):
    """A server SDK has no participant identity to trust."""
    bus.receive(packet({"by_id": "u-alex"}), None, "agent.ask")
    await settle()

    assert received == []


async def test_the_agents_own_identity_is_never_accepted_as_a_sender(bus, received):
    bus.receive(packet({"snippet_id": None}), AGENT_PARTICIPANT_ID, "stage")
    await settle()

    assert received == []


async def test_a_stage_payload_is_accepted_from_a_participant(bus, received):
    bus.receive(packet({"snippet_id": "s-1"}), "u-alex", "stage")
    await settle()

    assert received == [(Topic.STAGE, StagePayload(snippet_id="s-1"), "u-alex")]


async def test_a_failing_handler_does_not_stop_later_packets(bus, participant):
    seen = []

    async def handler(payload, sender):
        seen.append(payload.by_id)
        if len(seen) == 1:
            raise RuntimeError("boom")

    bus.subscribe(Topic.ASK, handler)
    bus.receive(packet({"by_id": "u-alex"}), "u-alex", "agent.ask")
    bus.receive(packet({"by_id": "u-sarah"}), "u-sarah", "agent.ask")
    await settle()

    assert seen == ["u-alex", "u-sarah"]
