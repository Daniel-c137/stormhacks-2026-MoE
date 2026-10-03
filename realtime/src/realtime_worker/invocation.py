"""Turns deliberate requests into Invocations: the assistant's name in a final segment, the Ask
button, or a public @mention. Plain text matching, never an LLM.

Everything else is transcribed and saved but never sent to the brain for reasoning.
"""

import re
from dataclasses import dataclass
from uuid import uuid4

from contracts import AGENT_PARTICIPANT_ID, ChatMessage, Invocation, TranscriptSegment
from contracts.agent import InvocationVia

# Saying only the name makes the same speaker's next final segment the question, this long.
NAME_ONLY_SECONDS = 15.0
# After the Ask button, the presser's next final segment is the question, this long.
ASK_SECONDS = 30.0

LEADING_PUNCTUATION = " \t,.:;!?-" + chr(0x2013) + chr(0x2014)  # en and em dash


def default_aliases(agent_name: str) -> list[str]:
    """The name as written, plus its words apart: "OmniMan" -> ["OmniMan", "Omni Man"]."""
    apart = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", agent_name)
    return [agent_name] if apart == agent_name else [agent_name, apart]


def alias_pattern(aliases: list[str]) -> str:
    """Whole-word match for any alias, with the words written together or apart
    ("omniman", "Omni Man", "Omni-Man")."""
    variants = {
        tuple(w.casefold() for w in re.sub(r"(?<=[a-z])(?=[A-Z])", " ", a).split()) for a in aliases
    }
    sequences = sorted(
        (r"[\s.,-]*".join(re.escape(w) for w in words) for words in variants if words),
        key=len,
        reverse=True,
    )
    return "(?:" + "|".join(sequences) + ")"


@dataclass
class Pending:
    via: InvocationVia
    until: float


class WakeDetector:
    def __init__(self, aliases: list[str]):
        if not aliases:
            raise ValueError("WakeDetector needs at least one alias")
        name = alias_pattern(aliases)
        self._spoken = re.compile(rf"(?<!\w){name}(?!\w)", re.IGNORECASE)
        self._mention = re.compile(rf"(?<!\w)@{name}(?!\w)", re.IGNORECASE)
        self._pending: dict[str, Pending] = {}

    def arm_ask(self, speaker_id: str, at: float) -> None:
        """The Ask button: this speaker's next final segment is the question."""
        self._pending[speaker_id] = Pending("ask", at + ASK_SECONDS)

    def on_segment(self, segment: TranscriptSegment) -> Invocation | None:
        if not segment.is_final or segment.speaker_id == AGENT_PARTICIPANT_ID:
            return None
        pending = self._pending.pop(segment.speaker_id, None)
        if pending and segment.t_start > pending.until:
            pending = None

        match = self._spoken.search(segment.text)
        if pending:
            question = after(segment.text, match) if match else segment.text.strip()
            if not question:
                self._pending[segment.speaker_id] = pending
                return None
            return self._invocation(segment, pending.via, question)
        if match:
            question = after(segment.text, match)
            if not question:
                self._pending[segment.speaker_id] = Pending(
                    "voice", segment.t_end + NAME_ONLY_SECONDS
                )
                return None
            return self._invocation(segment, "voice", question)
        return None

    def on_chat(self, message: ChatMessage) -> Invocation | None:
        """Public @mentions only. Private questions reach the brain over HTTP, never the room."""
        if message.visibility != "public" or message.is_agent:
            return None
        if not self._mention.search(message.text):
            return None
        question = self._mention.sub(" ", message.text)
        question = re.sub(r"\s+([?.!,;:])", r"\1", re.sub(r"\s+", " ", question))
        question = question.strip().lstrip(LEADING_PUNCTUATION).strip()
        if not question:
            return None
        return Invocation(
            id=str(uuid4()),
            meeting_id=message.meeting_id,
            via="chat",
            visibility="public",
            asked_by_id=message.sender_id,
            asked_by_name=message.sender_name,
            question=question,
        )

    def _invocation(
        self, segment: TranscriptSegment, via: InvocationVia, question: str
    ) -> Invocation:
        return Invocation(
            id=str(uuid4()),
            meeting_id=segment.meeting_id,
            via=via,
            visibility="public",
            asked_by_id=segment.speaker_id,
            asked_by_name=segment.speaker_name,
            question=question,
            t=segment.t_start,
        )


def after(text: str, match: re.Match[str]) -> str:
    return text[match.end() :].lstrip(LEADING_PUNCTUATION).strip()
