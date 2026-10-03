from collections.abc import Awaitable, Callable
from typing import Protocol

from pydantic import BaseModel

from contracts import Topic


class RoomBus(Protocol):
    """Typed LiveKit data messages; payload type per topic is in contracts.TOPIC_PAYLOADS."""

    async def publish(
        self, topic: Topic, payload: BaseModel, *, to: list[str] | None = None
    ) -> None:
        """to=None broadcasts to the room; private payloads always pass explicit identities."""
        ...

    def subscribe(self, topic: Topic, handler: Callable[[BaseModel, str], Awaitable[None]]) -> None:
        """handler(payload, sender_identity)."""
        ...
