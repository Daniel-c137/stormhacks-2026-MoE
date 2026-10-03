from typing import Protocol

from contracts import ChatMessage, Invocation, TranscriptSegment

# After an answer, follow-ups count as questions without the wake phrase for this long.
FOLLOWUP_SECONDS = 12


class InvocationDetector(Protocol):
    """Turns a wake phrase, a follow-up or a public @mention into an Invocation.

    Everything else is transcribed and saved but never sent to the brain for reasoning.
    """

    def on_segment(self, segment: TranscriptSegment) -> Invocation | None: ...

    def on_chat(self, message: ChatMessage) -> Invocation | None: ...

    def open_followup(self, speaker_id: str, at: float) -> None: ...
