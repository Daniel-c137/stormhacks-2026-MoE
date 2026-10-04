"""Who is in the meeting room, on the meeting's clock (seconds since it started), and who needs
catching up as they arrive.

Someone joining CATCH_UP_AFTER_SECONDS or more after the start is caught up from the start; someone
coming back after that long away is caught up on the time they were away. Never the agent, never
twice for one absence, never someone on time. The worker only knows what it has seen: people
already in the room when it (re)starts are present, with no catch-up owed for how they got there.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from contracts import AGENT_PARTICIPANT_ID

CATCH_UP_AFTER_SECONDS = 120.0


class Span(NamedTuple):
    """Meeting seconds someone missed."""

    since: float
    until: float


@dataclass
class Seen:
    first_seen: float  # when they joined, or the worker's start for those already here
    left_at: float | None = None  # None while they are in the room


class Presence:
    def __init__(self, after: float = CATCH_UP_AFTER_SECONDS):
        self._after = after
        self._seen: dict[str, Seen] = {}

    def already_here(self, identities: Iterable[str], now: float) -> None:
        """The people in the room when the worker joined it."""
        for identity in identities:
            self._seen.setdefault(identity, Seen(first_seen=now))

    def joined(self, identity: str, now: float) -> Span | None:
        """Someone came in: the span to catch them up on, if any."""
        if identity == AGENT_PARTICIPANT_ID:
            return None
        seen = self._seen.get(identity)
        if seen is None:
            self._seen[identity] = Seen(first_seen=now)
            return Span(0, now) if now >= self._after else None
        if seen.left_at is None:
            return None  # already here: a repeated event
        left_at, seen.left_at = seen.left_at, None
        return Span(left_at, now) if now - left_at >= self._after else None

    def left(self, identity: str, now: float) -> None:
        seen = self._seen.setdefault(identity, Seen(first_seen=now))
        if seen.left_at is None:
            seen.left_at = now

    def present(self, identity: str) -> bool:
        seen = self._seen.get(identity)
        return seen is not None and seen.left_at is None
