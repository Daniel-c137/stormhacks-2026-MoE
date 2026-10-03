from typing import Protocol

from contracts import Answer, ChatMessage, Invocation, TranscriptSegment


class BrainClient(Protocol):
    """HTTP client for the brain's /internal routes."""

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None: ...

    async def ingest_public_chat(self, meeting_id: str, message: ChatMessage) -> None: ...

    async def invoke(self, invocation: Invocation, recent: list[TranscriptSegment]) -> Answer: ...
