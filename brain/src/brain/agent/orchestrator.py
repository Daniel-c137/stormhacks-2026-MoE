from typing import Protocol

from contracts import Answer, Invocation, TranscriptSegment


class Orchestrator(Protocol):
    """Answers one deliberate invocation with typed tools.

    Cites evidence for every retrieved fact, separates facts from inference, and reports missing
    or failed sources in Answer.unavailable. Read-only tools only; no external writes.
    """

    async def answer(self, invocation: Invocation, recent: list[TranscriptSegment]) -> Answer: ...
