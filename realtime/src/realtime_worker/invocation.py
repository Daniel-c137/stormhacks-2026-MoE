"""Turns deliberate requests into Invocations: the assistant's name in a final segment, the Ask
button, or a public @mention. Plain text matching, never an LLM.

Everything else is transcribed and saved but never sent to the brain for reasoning.
"""

import math
import re
from dataclasses import dataclass, replace
from uuid import uuid4

from contracts import AGENT_PARTICIPANT_ID, ChatMessage, Invocation, TranscriptSegment
from contracts.agent import InvocationVia

# Saying only the name makes the same speaker's next final segment the question, this long after
# the name ends, unless someone else speaks first.
NAME_ONLY_SECONDS = 8.0
# After the Ask button, the presser's next final segment is the question, this long.
ASK_SECONDS = 30.0

ARABIC_COMMA = chr(0x060C)
# with en and em dashes, the ellipsis character and the Arabic comma
LEADING_PUNCTUATION = " \t,.:;!?-" + chr(0x2013) + chr(0x2014) + chr(0x2026) + ARABIC_COMMA
# Words people say before addressing someone: "Hey Polaris", "OK so, Polaris".
OPENERS = r"(?:(?:hey|hi|ok|okay|so|um|uh|hmm|alright|right|and)\W+)*"
# Where else in a segment someone may start addressing the assistant: a later sentence ("...we
# are covered there. Um, Polaris, ...") or a filler word mid-sentence ("covered there um,
# Polaris, ..."). Scribe keeps talk without a long pause in one segment.
LATER_STARTS = re.compile(
    r"(?<=[.!?\u2026\u061F])\s+|(?<!\w)(?=(?:um|uh|hmm|so|okay|ok|hey)\W)", re.IGNORECASE
)
# After the name, one of these, or a question or command word, says it is being addressed:
# "Polaris, ...", "Polaris what's ...". "Polaris said earlier" is talking about it.
SEPARATORS = ",:?!." + chr(0x2026) + ARABIC_COMMA
ASKING_WORDS = (
    "what whats who whose when where why how is are was were can could do does did should will "
    "would please tell show give find check list summarize summarise remind"
).split()
# These start a statement as often as a question ("Polaris is down again"), so after them the
# segment must ask: a question mark follows.
QUESTION_MARKS = "?" + chr(0x061F)  # and the Arabic question mark
AUXILIARIES = frozenset("is are was were can could do does did should will would".split())
# While waiting for the question, shorter segments are filler ("Um,", "So...").
MIN_QUESTION_WORDS = 2
# A spoken question that trails off waits this long after its segment ends for the same
# speaker's next final segment, then goes as it is.
TRAILING_SECONDS = 6.0
# A question ending on one of these words, or shorter than MIN_FINISHED_WORDS, is unfinished.
TRAILING_WORDS = frozenset("of the a an to for about on in with and or is are what's whats".split())
MIN_FINISHED_WORDS = 3
ELLIPSES = ("...", chr(0x2026))


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


@dataclass
class Held:
    """A spoken question that trailed off, waiting for the rest of the sentence. A fragment said
    after the name alone is never a question on its own: if nothing completes it, or someone
    else speaks, it is dropped."""

    via: InvocationVia
    question: str
    segment: TranscriptSegment  # where the question began: its speaker and time
    until: float
    fragment: bool = False


def fragment(text: str) -> bool:
    """Not a whole sentence: cut off ("Oh, I need-"), trailing off ("from..."), or under
    MIN_FINISHED_WORDS words without a question mark. Cut-off sounds ("w- b-") are not words."""
    text = text.strip()
    if text.endswith(("-", chr(0x2013), chr(0x2014), *ELLIPSES)):
        return True
    words = [w for w in text.split() if not w.rstrip(",.").endswith("-")]
    asks = any(mark in text for mark in QUESTION_MARKS)
    return not asks and len(words) < MIN_FINISHED_WORDS


def unfinished(question: str) -> bool:
    """It trails off ("what's the status of..."), ends on a joining word, or is too short to be
    a whole question. A question mark means the speaker finished."""
    question = question.strip()
    if question.endswith(ELLIPSES):
        return True
    if question.endswith("?"):
        return False
    words = question.split()
    if len(words) < MIN_FINISHED_WORDS:
        return True
    last = words[-1].strip(LEADING_PUNCTUATION).replace(chr(0x2019), "'").casefold()
    return last in TRAILING_WORDS


def joined(start: str, rest: str) -> str:
    """Joins "what's the status of..." and "DS-104 in Jira." into one question."""
    start = start.rstrip()
    for ellipsis in ELLIPSES:
        start = start.removesuffix(ellipsis)
    rest = rest.strip().lstrip("." + ELLIPSES[1]).strip()
    return f"{start.rstrip()} {rest}"


class WakeDetector:
    def __init__(self, aliases: list[str], *, trailing_seconds: float = TRAILING_SECONDS):
        if not aliases:
            raise ValueError("WakeDetector needs at least one alias")
        name = alias_pattern(aliases)
        asking = "|".join(ASKING_WORDS)
        # Addressed, not mentioned: the name opens the segment, followed by a separator, the
        # end, or a question or command word; or it closes a question after a comma.
        self._opening = re.compile(
            rf"^\W*{OPENERS}{name}(?=\s*[{SEPARATORS}]|\W*$|\s+(?P<word>{asking})(?!\w))",
            re.IGNORECASE,
        )
        self._closing = re.compile(
            rf"[,{ARABIC_COMMA}]\s*{name}[^\w?{QUESTION_MARKS}]*[?{QUESTION_MARKS}]\W*$",
            re.IGNORECASE,
        )
        self._name = re.compile(rf"(?<!\w){name}(?!\w)", re.IGNORECASE)
        self._openers = re.compile(rf"^\W*{OPENERS}", re.IGNORECASE)
        self._mention = re.compile(rf"(?<!\w)@{name}(?!\w)", re.IGNORECASE)
        self._pending: dict[str, Pending] = {}
        self._trailing_seconds = trailing_seconds
        self._held: dict[str, Held] = {}
        self._overdue: list[Held] = []

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

    def next_due(self) -> float | None:
        """When the earliest held question stops waiting, on the segments' clock; None if none
        is held. Whoever owns the clock calls due() then, so a held question is never lost."""
        if self._overdue:
            return -math.inf
        return min((h.until for h in self._held.values()), default=None)

    def due(self, now: float) -> list[Invocation]:
        """Held questions whose wait is over by `now`, sent as they are. A fragment after the
        name alone that nothing completed is dropped."""
        overdue, self._overdue = self._overdue, []
        for speaker_id, held in list(self._held.items()):
            if held.until <= now:
                overdue.append(self._held.pop(speaker_id))
        return [self._invocation(h.segment, h.via, h.question) for h in overdue if not h.fragment]

    def on_segment(self, segment: TranscriptSegment) -> Invocation | None:
        if not segment.is_final or segment.speaker_id == AGENT_PARTICIPANT_ID:
            return None
        speaker = segment.speaker_id
        self._others_spoke(speaker)
        addressed, question = self._addressed(segment.text)
        # Saying the name without addressing it is talking about the assistant, not to it.
        about = not addressed and bool(self._name.search(segment.text))

        if held := self._held.pop(speaker, None):
            if segment.t_start > held.until:
                if not held.fragment:
                    self._overdue.append(held)  # too late to join; due() sends it as it is
            elif held.fragment and about:
                return None
            elif not addressed:  # saying the name again starts over
                question = joined(held.question, segment.text)
                if held.fragment and fragment(segment.text):
                    until = max(held.until, segment.t_end + self._trailing_seconds)
                    self._held[speaker] = replace(held, question=question, until=until)
                    return None
                return self._spoken(held.segment, held.via, question, heard_until=segment.t_end)

        pending = self._pending.pop(speaker, None)
        if pending and segment.t_start > pending.until:
            pending = None

        if pending and pending.via == "voice":
            if about:
                return None
            if addressed and not question:  # the name alone again: the wait starts over
                pending = None
        if pending:
            if not addressed:
                question = segment.text.strip()
            if len(question.split()) < MIN_QUESTION_WORDS:
                self._pending[speaker] = pending
                return None
            if pending.via == "ask":
                return self._invocation(segment, pending.via, question)
            if not addressed and fragment(question):
                until = max(pending.until, segment.t_end + self._trailing_seconds)
                self._held[speaker] = Held("voice", question, segment, until, fragment=True)
                return None
            return self._spoken(segment, pending.via, question, heard_until=segment.t_end)
        if not addressed:
            return None
        if not question:
            self._pending[speaker] = Pending("voice", segment.t_end + NAME_ONLY_SECONDS)
            return None
        return self._spoken(segment, "voice", question, heard_until=segment.t_end)

    def _others_spoke(self, speaker_id: str) -> None:
        """Someone else's final segment ends everyone else's wait after the name alone, and
        drops a fragment waiting there. The Ask button and a started question still wait."""
        for other, pending in list(self._pending.items()):
            if other != speaker_id and pending.via == "voice":
                del self._pending[other]
        for other, held in list(self._held.items()):
            if other != speaker_id and held.fragment:
                del self._held[other]

    def _spoken(
        self, segment: TranscriptSegment, via: InvocationVia, question: str, *, heard_until: float
    ) -> Invocation | None:
        """A spoken question goes at once, or waits for the rest of the sentence if it trails
        off. `segment` is where the question began; `heard_until` is when its last part ended."""
        if unfinished(question):
            until = heard_until + self._trailing_seconds
            self._held[segment.speaker_id] = Held(via, question, segment, until)
            return None
        return self._invocation(segment, via, question)

    def _addressed(self, text: str) -> tuple[bool, str]:
        """(is the assistant addressed, the question). "Polaris, X", "Polaris what X" and
        "X, Polaris?" ask X, whether the name opens the segment, a later sentence in it, or
        follows a filler word ("...covered there. Um, Polaris, X"); the question is the rest of
        the segment. Any other mention is talking about the assistant, not to it."""
        name_alone = False
        for start in self._starts(text):
            rest = text[start:]
            opening = self._opening.search(rest)
            if opening and (word := opening.group("word")) and word.casefold() in AUXILIARIES:
                if not any(mark in rest[opening.end() :] for mark in QUESTION_MARKS):
                    opening = None  # "Polaris is down again"
            if opening:
                if question := after(rest, opening):
                    return True, question
                name_alone = True
        if closing := self._closing.search(text):
            before = self._openers.sub("", text[: closing.start()])
            before = before.rstrip(LEADING_PUNCTUATION).strip()
            if before:
                return True, before + "?"
        return name_alone, ""

    @staticmethod
    def _starts(text: str) -> list[int]:
        """Where addressing may begin: the segment's start, then each later sentence or filler."""
        return [0, *(m.end() for m in LATER_STARTS.finditer(text) if m.end() > 0)]

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
