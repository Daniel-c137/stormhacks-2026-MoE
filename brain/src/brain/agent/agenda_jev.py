"""Labelling the live transcript against the agenda with Jev (JEV_MODEL), in place of Gemini.

One request per stretch asks which agenda item each new line is about (a choice among the items'
labels, "other" for the team's work off the agenda and "none" for small talk) and, for each
pending item, how likely its discussion is over by the end of the stretch (a noul). The lines just
before the stretch are given as context. A long backlog is asked about in parts of JEV_MAX_LINES
lines, each part seeing the lines before it.

timekeeping.track_agenda decides what the answers change, exactly as it does with Gemini's.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from contracts import AgendaItem, TranscriptSegment

from .ask import clip

NO_ITEM = "none"  # timekeeping's label for talk about no agenda item: here, small talk
OTHER = "other"  # the team's work, off the agenda: about no item, but not small talk either
JEV_MAX_LINES = 40  # new lines per request; Jev answered 88 questions in one in 350 ms
CONTEXT_LINES = 12  # lines before a part, given as context
# How likely an item must be over for it to be ticked. With OVER's wording Jev put work still
# planned or in progress ("I'll cut the link today") at 0.25 or less, and reported results and
# closing lines at 0.6 to 0.95; a terse "Yes, agreed." can fall short and is then ticked at the
# next line, still timed at its last word.
COVERED_P = 0.5

OVER = {
    "true": "It was decided or answered, someone reported it done or gave its result, someone "
    "said it is closed, or it was really discussed and the talk has now moved on to another "
    "topic.",
    "false": "It is still being weighed, work on it is still in progress or only planned, only a "
    "side point was settled, it was only named in passing or put off for later, or nobody "
    "discussed it.",
}


class Jev(Protocol):
    model: str

    async def decide(
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]: ...


async def jev_labels(
    jev: Jev,
    labels: Mapping[str, AgendaItem],
    current: str,
    segments: Sequence[TranscriptSegment],
    earlier: Sequence[TranscriptSegment] = (),
) -> tuple[list[str], set[str]]:
    """Each segment's label ("a1", ..., "other" or "none"), and the labels of the pending items
    Jev finds over. `current` is the label of the item being discussed before the stretch."""
    pending = [label for label, item in labels.items() if item.status == "pending"]
    about: list[str] = []
    over: set[str] = set()
    context = list(earlier)
    for start in range(0, len(segments), JEV_MAX_LINES):
        part = segments[start : start + JEV_MAX_LINES]
        answers = await jev.decide(
            state(labels, current, part, context[-CONTEXT_LINES:]),
            questions(labels, pending, len(part)),
        )
        said = [answers[f"about_{n}"]["choice"] for n in range(1, len(part) + 1)]
        about += said
        over |= {label for label in pending if answers[f"closed_{label}"]["noul"] >= COVERED_P}
        context += part
        current = next((label for label in reversed(said) if label in labels), current)
    return about, over


def state(
    labels: Mapping[str, AgendaItem],
    current: str,
    part: Sequence[TranscriptSegment],
    earlier: Sequence[TranscriptSegment],
) -> dict[str, Any]:
    return {
        "agenda": {label: describe(item) for label, item in labels.items()},
        "being_discussed_before": current,
        "earlier_lines": [line(s) for s in earlier],
        "new_lines": {str(n): line(s) for n, s in enumerate(part, 1)},
    }


def questions(
    labels: Mapping[str, AgendaItem], pending: Sequence[str], lines: int
) -> dict[str, dict[str, Any]]:
    options = {label: item.title for label, item in labels.items()}
    options[OTHER] = "The team's work, but a topic that is not on the agenda."
    options[NO_ITEM] = "Small talk, greetings, setup or filler: not about the team's work."
    asked: dict[str, dict[str, Any]] = {
        f"about_{n}": {
            "type": "choice",
            "instructions": f"Which agenda item is new line {n} about?",
            "criteria": options,
        }
        for n in range(1, lines + 1)
    }
    for label in pending:
        asked[f"closed_{label}"] = {
            "type": "noul",
            "instructions": "By the end of the new lines, is the team's discussion of the "
            f"agenda item '{labels[label].title}' over?",
            "criteria": OVER,
        }
    return asked


def describe(item: AgendaItem) -> str:
    timebox = f", {item.minutes} min" if item.minutes else ""
    return f"{item.title} ({item.status}{timebox}, {discussed(item.discussed_s)})"


def discussed(seconds: float) -> str:
    if seconds < 1:
        return "not discussed yet"
    minutes, rest = divmod(int(seconds), 60)
    parts = (f"{minutes} min" if minutes else "", f"{rest} s" if rest else "")
    return "discussed " + " ".join(p for p in parts if p)


def line(segment: TranscriptSegment) -> str:
    minutes, seconds = divmod(int(segment.t_start), 60)
    return f"[{minutes:02d}:{seconds:02d}] {segment.speaker_name}: {clip(segment.text)}"
