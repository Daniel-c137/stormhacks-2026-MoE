"""Catching up late joiners: who is in the room on the meeting's clock, and the one private chat
message from Polaris that someone joining 5 minutes or more late (or coming back after 5 minutes
or more away) gets. Never the agent, never twice for one absence, never someone on time, never the
people already there when the worker (re)starts; never broadcast, spoken or stored."""

import asyncio
import logging
from dataclasses import dataclass

import pytest

from contracts import AGENT_PARTICIPANT_ID, CatchUpResponse, ChatMessage, Topic
from realtime_worker.brain_client import BrainUnavailable
from realtime_worker.meeting_agent import MeetingAgent
from realtime_worker.presence import CATCH_UP_AFTER_SECONDS, Presence, Span
from realtime_worker.state import RoomAgentState
from realtime_worker.worker import watch_presence

pytestmark = pytest.mark.anyio

MEETING = "m-1"
CAUGHT_UP = "Catching you up (00:00 to 07:00):\n- Refunds ship Friday. (01:30)\nAsk me privately."


# presence on its own


def test_catching_up_starts_five_minutes_in():
    assert CATCH_UP_AFTER_SECONDS == 300


def test_an_on_time_joiner_is_not_caught_up_and_a_late_one_is_from_the_start():
    presence = Presence()

    assert presence.joined("u-alex", 60) is None
    assert presence.joined("u-sarah", 299.9) is None
    assert presence.joined("u-bob", 300) == Span(0, 300)
    assert presence.joined("u-eve", 1200) == Span(0, 1200)
    assert all(presence.present(p) for p in ("u-alex", "u-sarah", "u-bob", "u-eve"))


def test_a_rejoin_after_five_minutes_away_covers_only_the_absence_once():
    presence = Presence()
    presence.joined("u-alex", 10)

    presence.left("u-alex", 400)
    assert not presence.present("u-alex")
    assert presence.joined("u-alex", 700) == Span(400, 700)
    assert presence.joined("u-alex", 705) is None  # a repeated join event: same absence


def test_a_short_blip_is_not_an_absence_worth_catching_up_on():
    presence = Presence()
    presence.joined("u-alex", 10)

    presence.left("u-alex", 400)

    assert presence.joined("u-alex", 699) is None
    assert presence.present("u-alex")


def test_a_late_joiner_who_drops_out_briefly_is_not_caught_up_again():
    presence = Presence()
    assert presence.joined("u-sarah", 420) == Span(0, 420)

    presence.left("u-sarah", 430)

    assert presence.joined("u-sarah", 450) is None


def test_people_already_there_when_the_worker_starts_are_never_caught_up_for_it():
    presence = Presence()
    presence.already_here(["u-alex", "u-sarah"], 900)

    assert presence.joined("u-alex", 905) is None  # a late join event for someone already in
    assert presence.joined("u-bob", 910) == Span(0, 910)  # joined after the restart
    presence.left("u-sarah", 1000)
    assert presence.joined("u-sarah", 1400) == Span(1000, 1400)  # a later absence still counts


def test_the_agent_is_never_caught_up():
    presence = Presence()
    presence.already_here([AGENT_PARTICIPANT_ID], 0)

    assert presence.joined(AGENT_PARTICIPANT_ID, 900) is None
    presence.left(AGENT_PARTICIPANT_ID, 1000)
    assert presence.joined(AGENT_PARTICIPANT_ID, 1400) is None


# the agent sending it


class Clock:
    def __init__(self, now: float = 0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeBus:
    def __init__(self):
        self.published: list[tuple[Topic, object, list[str] | None]] = []

    async def publish(self, topic, payload, *, to=None) -> None:
        self.published.append((topic, payload, to))

    def subscribe(self, topic, handler) -> None:
        pass

    def on(self, topic: Topic) -> list:
        return [(p, to) for t, p, to in self.published if t == topic]


class FakeBrain:
    def __init__(self):
        self.catch_ups: list[tuple[str, str, float, float]] = []
        self.result: CatchUpResponse | Exception = CatchUpResponse(
            text=CAUGHT_UP, source_times=[90]
        )
        self.chat: list[ChatMessage] = []

    async def catch_up(
        self, meeting_id: str, participant_id: str, since: float, until: float
    ) -> CatchUpResponse:
        self.catch_ups.append((meeting_id, participant_id, since, until))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def ingest_public_chat(self, meeting_id: str, message: ChatMessage) -> None:
        self.chat.append(message)

    async def agent_joined(self, meeting_id: str) -> None:
        pass


class Unused:
    """TTS, speaker and public chat: a catch-up never uses them."""

    def __init__(self):
        self.calls: list[str] = []

    def synthesize(self, text: str, voice_id: str):
        self.calls.append(text)
        raise AssertionError("a catch-up is never spoken")

    async def play(self, frames) -> None:
        raise AssertionError("a catch-up is never spoken")

    async def send(self, text: str) -> str:
        self.calls.append(text)
        raise AssertionError("a catch-up is never posted in public chat")


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def bus():
    return FakeBus()


@pytest.fixture
def brain():
    return FakeBrain()


@pytest.fixture
async def agent(bus, brain, clock):
    unused = Unused()
    agent = MeetingAgent(
        MEETING,
        bus=bus,
        brain=brain,
        state=RoomAgentState(bus),
        tts=unused,
        speaker=unused,
        chat=unused,
        agent_name="Polaris",
        default_voice_id=None,
        agenda_tick_seconds=3600,
        fact_check_tick_seconds=3600,
        clock=clock,
        catch_up_delay=0,
    )
    agent.start()
    yield agent
    await agent.aclose()


async def until(condition, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.001)


async def settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


def private_messages(bus: FakeBus) -> list[tuple[ChatMessage, list[str] | None]]:
    return bus.on(Topic.PRIVATE_CHAT)


async def test_an_on_time_joiner_gets_nothing(agent, bus, brain, clock):
    clock.now = 120

    agent.on_participant_joined("u-sarah")
    await settle()

    assert brain.catch_ups == []
    assert private_messages(bus) == []


async def test_a_late_joiner_gets_one_private_message_from_polaris_only(agent, bus, brain, clock):
    clock.now = 420

    agent.on_participant_joined("u-sarah")
    await until(lambda: private_messages(bus))
    await settle()

    assert brain.catch_ups == [(MEETING, "u-sarah", 0, 420)]
    [(message, to)] = private_messages(bus)
    assert to == ["u-sarah"]
    assert isinstance(message, ChatMessage)
    assert (message.sender_id, message.sender_name, message.is_agent) == (
        AGENT_PARTICIPANT_ID,
        "Polaris",
        True,
    )
    assert (message.visibility, message.recipient_id) == ("private", "u-sarah")
    assert (message.meeting_id, message.text) == (MEETING, CAUGHT_UP)
    # Nowhere else: not to the room, not saved, not spoken.
    assert [t for t, _, _ in bus.published if t != Topic.AGENT_STATE] == [Topic.PRIVATE_CHAT]
    assert brain.chat == []


async def test_a_rejoin_after_five_minutes_covers_only_what_they_missed(agent, bus, brain, clock):
    clock.now = 30
    agent.on_participant_joined("u-alex")
    clock.now = 400
    agent.on_participant_left("u-alex")
    clock.now = 760

    agent.on_participant_joined("u-alex")
    await until(lambda: private_messages(bus))

    assert brain.catch_ups == [(MEETING, "u-alex", 400, 760)]
    [(_, to)] = private_messages(bus)
    assert to == ["u-alex"]


async def test_a_short_blip_gets_nothing(agent, bus, brain, clock):
    clock.now = 30
    agent.on_participant_joined("u-alex")
    clock.now = 400
    agent.on_participant_left("u-alex")
    clock.now = 450

    agent.on_participant_joined("u-alex")
    await settle()

    assert brain.catch_ups == []
    assert private_messages(bus) == []


async def test_the_same_absence_is_never_caught_up_twice(agent, bus, brain, clock):
    clock.now = 420
    agent.on_participant_joined("u-sarah")
    agent.on_participant_joined("u-sarah")  # LiveKit said so twice
    await until(lambda: private_messages(bus))
    await settle()

    assert len(brain.catch_ups) == 1
    assert len(private_messages(bus)) == 1


async def test_a_restarted_worker_does_not_catch_up_people_already_in_the_room(
    agent, bus, brain, clock
):
    clock.now = 900
    agent.already_here(["u-alex", "u-sarah", AGENT_PARTICIPANT_ID])
    agent.on_participant_joined("u-alex")
    await settle()
    assert brain.catch_ups == []

    clock.now = 910
    agent.on_participant_joined("u-bob")  # joins after the restart
    await until(lambda: private_messages(bus))

    assert brain.catch_ups == [(MEETING, "u-bob", 0, 910)]


async def test_the_agent_itself_is_never_caught_up(agent, bus, brain, clock):
    clock.now = 900

    agent.on_participant_joined(AGENT_PARTICIPANT_ID)
    await settle()

    assert brain.catch_ups == []


async def test_a_brain_failure_is_logged_and_nothing_is_sent(agent, bus, brain, clock, caplog):
    brain.result = BrainUnavailable("POST /internal/meetings/m-1/catch-up failed (HTTP 503)")
    clock.now = 420

    with caplog.at_level(logging.WARNING):
        agent.on_participant_joined("u-sarah")
        await until(lambda: brain.catch_ups)
        await settle()

    assert private_messages(bus) == []
    assert any("catch" in r.getMessage() and "u-sarah" in r.getMessage() for r in caplog.records)


async def test_nothing_to_say_sends_nothing(agent, bus, brain, clock):
    brain.result = CatchUpResponse(text=None)
    clock.now = 420

    agent.on_participant_joined("u-sarah")
    await until(lambda: brain.catch_ups)
    await settle()

    assert private_messages(bus) == []


async def test_someone_who_left_again_before_it_was_ready_gets_nothing(agent, bus, brain, clock):
    gate = asyncio.Event()
    answer = brain.catch_up

    async def slow(*args):
        await gate.wait()
        return await answer(*args)

    brain.catch_up = slow
    clock.now = 420
    agent.on_participant_joined("u-sarah")
    await settle()
    clock.now = 425
    agent.on_participant_left("u-sarah")

    gate.set()
    await until(lambda: brain.catch_ups)
    await settle()

    assert private_messages(bus) == []


async def test_closing_the_agent_cancels_a_pending_catch_up(bus, brain, clock):
    unused = Unused()
    agent = MeetingAgent(
        MEETING,
        bus=bus,
        brain=brain,
        state=RoomAgentState(bus),
        tts=unused,
        speaker=unused,
        chat=unused,
        agent_name="Polaris",
        default_voice_id=None,
        agenda_tick_seconds=3600,
        fact_check_tick_seconds=3600,
        clock=clock,
        catch_up_delay=3600,
    )
    agent.start()
    clock.now = 420
    agent.on_participant_joined("u-sarah")

    await agent.aclose()

    assert brain.catch_ups == []
    assert private_messages(bus) == []


# the worker's LiveKit wiring


@dataclass
class Remote:
    identity: str


class Room:
    def __init__(self, *present: str):
        self.remote_participants = {p: Remote(p) for p in present}
        self.handlers: dict[str, list] = {}

    def on(self, event: str, handler) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, participant: Remote) -> None:
        for handler in self.handlers.get(event, []):
            handler(participant)


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def already_here(self, identities) -> None:
        self.events.append(("here", sorted(identities)))

    def on_participant_joined(self, identity: str) -> None:
        self.events.append(("joined", identity))

    def on_participant_left(self, identity: str) -> None:
        self.events.append(("left", identity))


def test_the_worker_marks_who_is_already_there_then_follows_joins_and_leaves():
    room = Room("u-alex", "u-sarah")
    agent = Recorder()

    watch_presence(room, agent)  # type: ignore[arg-type]
    room.emit("participant_connected", Remote("u-bob"))
    room.emit("participant_disconnected", Remote("u-alex"))

    assert agent.events == [
        ("here", ["u-alex", "u-sarah"]),
        ("joined", "u-bob"),
        ("left", "u-alex"),
    ]
