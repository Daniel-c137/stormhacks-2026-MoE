"""RoomBus over LiveKit data packets: JSON payloads, reliable, one topic per payload type.

Anyone in the room can publish data, so what the worker accepts is narrow: only the topics the
board sends the worker, only from a participant, only a payload that validates, and the sender is
the identity LiveKit attached to the packet, never anything the payload says.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from contracts import AGENT_PARTICIPANT_ID, TOPIC_PAYLOADS, Topic

log = logging.getLogger(__name__)

# board -> realtime. Every other topic is the agent's to publish; a participant sending one is
# ignored.
BOARD_TOPICS = frozenset({Topic.RESPONSE_ACTION, Topic.ASK, Topic.STAGE})

Handler = Callable[[Any, str], Awaitable[None]]


class DataPublisher(Protocol):
    """rtc.LocalParticipant's publish_data."""

    async def publish_data(
        self,
        payload: bytes | str,
        *,
        reliable: bool = True,
        destination_identities: list[str] = ...,
        topic: str = "",
    ) -> None: ...


class LiveKitBus:
    def __init__(self, participant: DataPublisher):
        self._participant = participant
        self._handlers: dict[Topic, list[Handler]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def attach(self, room) -> None:
        """Receive the room's data packets (rtc.Room's data_received)."""

        def on_data(packet) -> None:
            sender = packet.participant.identity if packet.participant else None
            self.receive(packet.data, sender, packet.topic)

        room.on("data_received", on_data)

    async def publish(
        self, topic: Topic, payload: BaseModel, *, to: list[str] | None = None
    ) -> None:
        """to=None broadcasts to the room. An empty list is refused: LiveKit would send it to
        everyone, and a list is only ever given for something private."""
        expected = TOPIC_PAYLOADS[topic]
        if not isinstance(payload, expected):
            raise TypeError(f"{topic} carries {expected.__name__}, not {type(payload).__name__}")
        if to is not None and not to:
            raise ValueError(f"No recipients for a private {topic} payload")
        await self._participant.publish_data(
            payload.model_dump_json().encode(),
            reliable=True,
            destination_identities=list(to or []),
            topic=topic.value,
        )

    def subscribe(self, topic: Topic, handler: Handler) -> None:
        """handler(payload, sender_identity), the identity LiveKit verified."""
        self._handlers.setdefault(topic, []).append(handler)

    def receive(self, data: bytes, sender: str | None, topic: str | None) -> None:
        """One data packet: dropped unless a participant sent a board topic that validates."""
        if not sender or sender == AGENT_PARTICIPANT_ID or topic is None:
            return
        try:
            known = Topic(topic)
        except ValueError:
            return
        if known not in BOARD_TOPICS:
            log.warning("Ignoring %s from participant %s: only the agent sends it", topic, sender)
            return
        try:
            payload = TOPIC_PAYLOADS[known].model_validate_json(data)
        except ValidationError:
            log.warning("Ignoring an invalid %s payload from %s", topic, sender)
            return
        for handler in self._handlers.get(known, []):
            task = asyncio.get_running_loop().create_task(self._run(handler, payload, sender))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    @staticmethod
    async def _run(handler: Handler, payload: BaseModel, sender: str) -> None:
        try:
            await handler(payload, sender)
        except Exception:
            log.exception("Handling a data packet from %s failed", sender)
