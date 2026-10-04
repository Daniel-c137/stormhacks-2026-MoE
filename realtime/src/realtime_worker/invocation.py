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
# Words people say before addressing someone: "Hey Polaris", "OK so, Polaris".
OPENERS = r"(?:(?:hey|hi|ok|okay|so|um|uh|alright|right|and)\W+)*"
# While waiting for the question, shorter segments are filler ("Um,", "So...").
MIN_QUESTION_WORDS = 2


def default_aliases(agent_name: str) -> list[str]:
    """The name as written, plus its words apart when it has several: "Polaris" -> ["Polaris"],
    "NorthStar" -> ["NorthStar", "North Star"]."""
    apart = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", agent_name)
    return [agent_name] if apart == agent_name else [agent_name, apart]


def alias_pattern(aliases: list[str]) -> str:
    """Any alias, with the words written together or apart ("northstar", "North Star",
    "North-Star"). Only spaces and hyphens may sit between the words, never a full stop."""
    variants = {
        tuple(w.casefold() for w in re.sub(r"(?<=[a-z])(?=[A-Z])", " ", a).split()) for a in aliases
    }
    sequences = sorted(
        (r"[\s-]*".join(re.escape(w) for w in words) for words in variants if words),
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
        # Addressed, not mentioned: the name opens the segment or closes it.
        self._opening = re.compile(rf"^\W*{OPENERS}{name}(?!\w)", re.IGNORECASE)
        self._closing = re.compile(rf"(?<!\w){name}\W*$", re.IGNORECASE)
        self._openers = re.compile(rf"^\W*{OPENERS}", re.IGNORECASE)
        self._mention = re.compile(rf"(?<!\w)@{name}(?!\w)", re.IGNORECASE)
        self._pending: dict[str, Pending] = {}

    def arm_ask(self, speaker_id: str, at: float) -> None:
        """The Ask button: this speaker's next final segment is the question.

        `at` is on the segments' clock: seconds from the meeting start, like t_start."""
        self._pending[speaker_id] = Pending("ask", at + ASK_SECONDS)

    def cancel_ask(self, speaker_id: str) -> bool:
        """Withdraw this speaker's Ask press. True if one was waiting; a name said alone is not
        an Ask press and stays."""
        pending = self._pending.get(speaker_id)
        if pending is None or pending.via != "ask":
            return False
        del self._pending[speaker_id]
        return True

    def on_segment(self, segment: TranscriptSegment) -> Invocation | None:
        if not segment.is_final or segment.speaker_id == AGENT_PARTICIPANT_ID:
            return None
        pending = self._pending.pop(segment.speaker_id, None)
        if pending and segment.t_start > pending.until:
            pending = None

        addressed, question = self._addressed(segment.text)
        if pending:
            if not addressed:
                question = segment.text.strip()
            if len(question.split()) < MIN_QUESTION_WORDS:
                self._pending[segment.speaker_id] = pending
                return None
            return self._invocation(segment, pending.via, question)
        if not addressed:
            return None
        if not question:
            self._pending[segment.speaker_id] = Pending("voice", segment.t_end + NAME_ONLY_SECONDS)
            return None
        return self._invocation(segment, "voice", question)

    def _addressed(self, text: str) -> tuple[bool, str]:
        """(is the assistant addressed, the question). "Polaris, X" and "X, Polaris?" ask X;
        a mid-sentence mention is talking about the assistant, not to it."""
        if opening := self._opening.search(text):
            if question := after(text, opening):
                return True, question
        if closing := self._closing.search(text):
            before = self._openers.sub("", text[: closing.start()])
            before = before.rstrip(LEADING_PUNCTUATION).strip()
            if before:
                mark = "?" if "?" in text[closing.end() - 1 :] else ""
                return True, before + mark
        return bool(opening), ""

    def on_chat(
        self, message: ChatMessage, *, sender_id: str, sender_name: str
    ) -> Invocation | None:
        """Public @mentions only. Private questions reach the brain over HTTP, never the room.

        sender_id and sender_name are the participant LiveKit verified, never the payload's
        claims: anyone can publish data messages. A payload naming someone else is refused."""
        if message.visibility != "public" or message.is_agent:
            return None
        if message.sender_id != sender_id:
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
            asked_by_id=sender_id,
            asked_by_name=sender_name,
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
