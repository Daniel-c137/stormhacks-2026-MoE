"""A private catch-up for someone who joined a live meeting late or came back after a while away.

One model call over the final segments of the span they missed and the meeting's agenda with its
status. Only what people said counts: never the agent's own lines, never chat. Each item the model
writes names the numbered lines it rests on; an item that cites none is dropped, and the time shown
is that of the first line it cites; a main point citing only what a decision or a for-you item
cites repeats it and is dropped. Where the meeting is now comes from the agenda's current item
when there is one. With too little said (under MIN_SENTENCES whole sentences) there is nothing to
send and no model call. Nothing here is stored or logged.
"""

import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

from pydantic import BaseModel, Field

from brain.llm import LLM
from brain.report.extraction import by_agent, clock
from brain.store import NotFound, Store
from contracts import (
    Agenda,
    CatchUpRequest,
    CatchUpResponse,
    Meeting,
    TranscriptSegment,
    get_identity,
)

from .ask import DATA_RULE, MARKERS, clip, fenced, fragment, oneline, render_agenda

MIN_SENTENCES = 3
MAX_WORDS = 120  # the whole message
MAX_ITEM_WORDS = 25
MAX_TITLE_WORDS = 12
MAX_LINES = 300  # the newest lines of a long span
MAX_POINTS = 3
MAX_DECISIONS = 3
MAX_FOR_YOU = 2
SENTENCE_ENDS = re.compile(r"(?<=[.!?؟])\s+")


class Point(BaseModel):
    text: str = Field(
        description="One short sentence of at most 20 words, using only facts from the lines "
        "it cites."
    )
    evidence_ids: list[str] = Field(
        default=[], description="Ids of the transcript lines it rests on, e.g. ['e3']."
    )


class CatchUpDraft(BaseModel):
    now: Point | None = Field(
        default=None,
        description="What the meeting is discussing at the end of the transcript, in a few "
        "words. Null when the agenda shows a current item, or when it is unclear.",
    )
    points: list[Point] = Field(
        default=[], description=f"The main points, at most {MAX_POINTS}, most important first."
    )
    decisions: list[Point] = Field(
        default=[], description="What people decided or agreed. Empty when nothing was."
    )
    for_you: list[Point] = Field(
        default=[],
        description="What was asked of, or said about, the person catching up, by name. Empty "
        "when nothing was.",
    )


class Item(NamedTuple):
    text: str
    times: list[float]  # of the lines it cites, oldest first


async def catch_up(
    store: Store, make_llm: Callable[[], LLM], meeting: Meeting, request: CatchUpRequest
) -> CatchUpResponse:
    """The message for request.participant_id about what was said between since and until. The
    model is made only when there is something to summarise."""
    said = missed(await store.transcript(meeting.id), request.since, request.until)
    if whole_sentences(said) < MIN_SENTENCES:
        return CatchUpResponse()
    lines = {f"e{n}": s for n, s in enumerate(said[-MAX_LINES:], start=1)}
    agenda = await store.agenda(meeting.id)
    name = await joiner_name(store, request.participant_id)
    draft = await make_llm().generate_structured(
        render_prompt(meeting, name, request, lines, agenda),
        CatchUpDraft,
        system=system_prompt(name, rejoined=request.since > 0),
    )
    return finish(draft, lines, agenda, request)


def missed(
    transcript: Sequence[TranscriptSegment], since: float, until: float
) -> list[TranscriptSegment]:
    """What people said from since to until, final and whole: not the agent, not fragments."""
    return [
        s
        for s in transcript
        if s.is_final and since <= s.t_start <= until and not by_agent(s) and not fragment(s.text)
    ]


def whole_sentences(said: Sequence[TranscriptSegment]) -> int:
    return sum(
        1
        for s in said
        for sentence in SENTENCE_ENDS.split(s.text.strip())
        if sentence.strip() and not fragment(sentence)
    )


async def joiner_name(store: Store, participant_id: str) -> str:
    try:
        person = await store.person(participant_id)
    except NotFound:
        return participant_id
    if person.short and person.short != person.name:
        return f"{person.name} (also called {person.short})"
    return person.name


def finish(
    draft: CatchUpDraft,
    lines: dict[str, TranscriptSegment],
    agenda: Agenda | None,
    request: CatchUpRequest,
) -> CatchUpResponse:
    """Only grounded items, at most MAX_WORDS in all; nothing to send when none is grounded."""
    decisions = grounded(draft.decisions, lines)[:MAX_DECISIONS]
    for_you = grounded(draft.for_you, lines)[:MAX_FOR_YOU]
    # A main point resting only on lines a decision or a for-you item cites repeats it.
    told = {t for item in [*decisions, *for_you] for t in item.times}
    points = [p for p in grounded(draft.points, lines) if not set(p.times) <= told][:MAX_POINTS]
    current = agenda_now(agenda)
    said_now = [] if current or draft.now is None else grounded([draft.now], lines)
    if not (points or decisions or for_you or said_now):
        return CatchUpResponse()
    now = [f"Now: {current}"] if current else [f"Now: {timed(item)}" for item in said_now]

    def message() -> list[str]:
        return [
            header(request),
            *now,
            *(f"- {timed(item)}" for item in points),
            *(f"Decided: {timed(item)}" for item in decisions),
            *(f"For you: {timed(item)}" for item in for_you),
            f"Ask {get_identity().agent_name} here, privately, for more.",
        ]

    # Over the limit, the main points go first, then decisions; what concerns them goes last.
    text = message()
    while words(text) > MAX_WORDS and (points or decisions or len(for_you) > 1):
        (points or decisions or for_you).pop()
        text = message()
    times = sorted({t for item in [*said_now, *points, *decisions, *for_you] for t in item.times})
    return CatchUpResponse(text="\n".join(text), source_times=times)


def grounded(points: Sequence[Point], lines: dict[str, TranscriptSegment]) -> list[Item]:
    """Items citing at least one line that exists, with the marker-free, word-capped text."""
    items = []
    for point in points:
        ids = dict.fromkeys(i.strip().strip("[]").lower() for i in point.evidence_ids)
        cited = sorted(lines[i].t_start for i in ids if i in lines)
        text = cap_words(oneline(MARKERS.sub("", point.text)), MAX_ITEM_WORDS)
        if cited and text:
            items.append(Item(text, cited))
    return items


def agenda_now(agenda: Agenda | None) -> str | None:
    """The agenda item being discussed now, and where it falls."""
    if agenda is None:
        return None
    for n, item in enumerate(agenda.items, start=1):
        if item.id == agenda.current_item_id and item.status == "pending":
            title = cap_words(oneline(item.title), MAX_TITLE_WORDS)
            return f"{title} (agenda item {n} of {len(agenda.items)})"
    return None


def header(request: CatchUpRequest) -> str:
    span = f"{clock(request.since)} to {clock(request.until)}"
    return f"While you were away ({span}):" if request.since > 0 else f"Catching you up ({span}):"


def timed(item: Item) -> str:
    return f"{item.text} ({clock(item.times[0])})"


def words(lines: Sequence[str]) -> int:
    return sum(len(line.split()) for line in lines)


def cap_words(text: str, limit: int) -> str:
    split = text.split()
    return text if len(split) <= limit else " ".join(split[:limit]) + "…"


# prompts


def system_prompt(name: str, *, rejoined: bool) -> str:
    agent = get_identity().agent_name
    who = oneline(name)
    arrived = (
        "has just come back to a meeting after being away"
        if rejoined
        else ("has just joined a meeting that is already under way")
    )
    return f"""You are {agent}, the assistant of a software team. {who} {arrived}. You are \
writing the notes for a short private catch-up that only they will see. Nobody asked you \
anything; you only summarise what they missed.

Rules:
- Use only the numbered transcript lines. Never add facts, names, dates, numbers or links that
  they do not contain.
- Every item lists in evidence_ids the ids of the lines it rests on. Never write ids, times or
  brackets in the text.
- points: the main points, at most {MAX_POINTS}, most important first.
- decisions: only what people clearly decided or agreed. Leave it empty rather than guess.
- for_you: only what was asked of, or said about, {who} by name. Leave it empty when nothing was.
- now: what the meeting is discussing at the end of the transcript, in a few words. Leave it
  null when the agenda shows an item as current.
- Do not repeat one fact in two places. Each item is one short sentence of at most 20 words.
  Address {who} as "you".
- Keep numbers, versions, issue keys and names exactly as the transcript writes them.
{DATA_RULE}
- Plain text, no markdown."""


def render_prompt(
    meeting: Meeting,
    name: str,
    request: CatchUpRequest,
    lines: dict[str, TranscriptSegment],
    agenda: Agenda | None,
) -> str:
    span = f"{clock(request.since)} to {clock(request.until)}"
    out = [
        f'Meeting: "{oneline(meeting.title)}"',
        f"Catching up: {oneline(name)} (id {oneline(request.participant_id)}), "
        f"who missed {span} of the meeting.",
        "",
    ]
    if agenda is not None and agenda.items:
        out.append("This meeting's agenda, with each item's status:")
        out += fenced([render_agenda(meeting, agenda)])
    else:
        out.append("This meeting has no agenda.")
    out += ["", "What people said in that time ([id] [time] speaker: text), oldest first:"]
    out += fenced(
        f"[{i}] [{clock(s.t_start)}] {s.speaker_name}: {clip(s.text)}" for i, s in lines.items()
    )
    return "\n".join(out)
