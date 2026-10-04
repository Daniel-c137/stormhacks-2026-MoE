"""Keeping time against the agenda during a live meeting.

The realtime worker calls this on a timer, never per utterance. Each tick labels the final
segments since the last tracked point with the items they are about in one model call, gives each
item the talk time of its own segments and marks items the team finished. The model also sees
the last few lines before the stretch and how long each item has been discussed, so it can tell
an item the talk has left from one still under way; an item discussed and then left for another
is covered even when the model does not say so. Nudges are decided by fixed rules, not the model,
and are only shown: the agent never speaks on its own.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from brain.llm import LLM, LLMError
from brain.store import Conflict, Store
from contracts import (
    AGENT_PARTICIPANT_ID,
    Agenda,
    AgendaItem,
    AgendaNudge,
    AgendaTrackResponse,
    Meeting,
    TranscriptSegment,
    get_identity,
)

from .ask import DATA_RULE, clip, fenced

SETTLE_S = 5.0  # segments ending this close to `now` wait a tick, so late finals are not skipped
PAUSE_S = 15.0  # a pause up to this long between utterances still counts as discussion
# With the worker's tick every 10 s, the sentence that finishes an item is asked about at the
# next tick after it settles (within about 15 s), and a one-word closer at the one after that.
# The worker's tick is what limits the model calls: at most one a tick, and none while nobody
# has said anything new.
MIN_TALK_S = 4.0  # the model is asked once the new stretch holds this much talk (a sentence),
MAX_WAIT_S = 15.0  # or once this long has passed since the tracked point
CONTEXT_SEGMENTS = 12  # lines before the stretch the model sees, to judge by but not to label
MOVED_ON_MIN_S = 20.0  # talk an item needs before leaving it for another item covers it
NOW_SLACK_S = 60.0  # how far a tick's `now` may run ahead of the brain's clock
END_WARN_MIN = 5  # nudge about items that have not come up this close to the scheduled end
MAX_BATCH = 200  # segments per classification; a longer backlog keeps the latest
SAVE_ATTEMPTS = 3  # saves of one tick's result before giving up on an agenda that keeps changing


NO_ITEM = "none"  # the label for talk about no agenda item


class TopicRun(BaseModel):
    first: int = Field(description="Number of the first segment of the run, e.g. 1.")
    last: int = Field(description="Number of the last segment of the run; first for one segment.")
    item: str = Field(
        description="Label of the agenda item these segments are about, e.g. 'a2', or 'none' for "
        "small talk, setup or a topic that is not on the agenda."
    )


class AgendaTrackDraft(BaseModel):
    topics: list[TopicRun] = Field(
        default=[],
        description="Every segment of the stretch, in order, as runs of consecutive segments "
        "about the same item.",
    )
    covered: list[str] = Field(
        default=[],
        description="Labels of items whose discussion is over by the end of this stretch: "
        "decided, answered, closed, or talked through and then left for something else. Not an "
        "item still being discussed, nor one only named in passing.",
    )


def track_system() -> str:
    agent = get_identity().agent_name
    return f"""You are {agent}. You keep time against a software team's meeting agenda. You read
the latest stretch of the live transcript and label each numbered segment with the agenda item it
is about.

Rules:
- "topics" labels every segment once, as runs of consecutive segments about the same thing: the
  first and last segment number and the item's label, or "{NO_ITEM}" for small talk, setup or a
  topic that is not on the agenda. A stretch often moves between items: label each segment by
  what it is about, not by the stretch as a whole. An item nobody discusses gets no segments.
- "covered" lists the items whose discussion is over by the end of this stretch. An item is over
  when the team decided or answered it, when someone says it is done, or when they talked it
  through and have moved on to something else. One clear sentence can settle an item ("the
  release moves to Monday").
- The lines under "Earlier" came just before the stretch. Do not label them; use them, and how
  long each item has been discussed, to judge. A short remark ("yes, agreed") is about what the
  lines before it are about. An item discussed there that the talk has left in this stretch is
  covered.
- An item still being discussed when the stretch ends is not covered yet. Neither is one only
  named in passing ("we'll get to the launch date later").
- Use only labels from the agenda, or "{NO_ITEM}". Judge only from the transcript.
{DATA_RULE}"""


def clock(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"


def discussed(seconds: float) -> str:
    """How long an item has been talked about so far, e.g. "discussed 1 min 35 s"."""
    if seconds < 1:
        return "not discussed yet"
    minutes, rest = divmod(int(seconds), 60)
    return "discussed " + " ".join(
        part for part in (f"{minutes} min" if minutes else "", f"{rest} s" if rest else "") if part
    )


def render_track_prompt(
    agenda: Agenda,
    labels: dict[str, AgendaItem],
    segments: Sequence[TranscriptSegment],
    earlier: Sequence[TranscriptSegment] = (),
) -> str:
    """The agenda with each item's state, the lines just before the stretch (unnumbered: context
    only) when there are any, and the stretch's segments, numbered to be labelled."""

    def describe(label: str, item: AgendaItem) -> str:
        timebox = f", {item.minutes} min" if item.minutes else ""
        return f"[{label}] {item.title} ({item.status}{timebox}, {discussed(item.discussed_s)})"

    current = next((lb for lb, i in labels.items() if i.id == agenda.current_item_id), NO_ITEM)
    context: list[str] = []
    if earlier:
        context = [
            "Earlier, just before this stretch, for context only ([mm:ss] speaker: text):",
            *fenced(f"[{clock(s.t_start)}] {s.speaker_name}: {clip(s.text)}" for s in earlier),
            "",
        ]
    return "\n".join(
        [
            "Agenda ([label] title (status, timebox, time discussed so far)):",
            *fenced(describe(label, item) for label, item in labels.items()),
            "",
            f"Being discussed before this stretch: {current}",
            "",
            *context,
            "Transcript stretch ([n] [mm:ss] speaker: text):",
            *fenced(
                f"[{n}] [{clock(s.t_start)}] {s.speaker_name}: {clip(s.text)}"
                for n, s in enumerate(segments, 1)
            ),
        ]
    )


@dataclass(frozen=True)
class Labelled:
    """What the model said about a stretch, checked against the agenda."""

    about: tuple[str | None, ...]  # per segment: its item's id, or None (no item, or no label)
    current: str | None  # the item of the last labelled segment
    covered: frozenset[str]


async def classify(
    llm: LLM,
    agenda: Agenda,
    segments: Sequence[TranscriptSegment],
    earlier: Sequence[TranscriptSegment] = (),
) -> Labelled:
    """Labels each segment (numbered from 1 in the order given) with an item; `earlier` is shown
    for context and not labelled. Labels not on the agenda and segment numbers outside the
    stretch are ignored; where runs overlap, the later one wins."""
    labels = {f"a{n}": item for n, item in enumerate(agenda.items, 1)}
    draft = await llm.generate_structured(
        render_track_prompt(agenda, labels, segments, earlier),
        AgendaTrackDraft,
        system=track_system(),
    )

    def item_id(label: str) -> str | None:
        item = labels.get(label.strip())
        return item.id if item else None

    marks: dict[int, str | None] = {}
    for run in draft.topics:
        about = item_id(run.item)
        if about is None and run.item.strip().lower() != NO_ITEM:
            continue
        for n in range(max(run.first, 1), min(run.last, len(segments)) + 1):
            marks[n] = about
    return Labelled(
        about=tuple(marks.get(n) for n in range(1, len(segments) + 1)),
        current=marks[max(marks)] if marks else None,
        covered=frozenset(i for i in map(item_id, draft.covered) if i),
    )


def runs(segments: Sequence[TranscriptSegment], until: float) -> list[tuple[float, float]]:
    """Stretches of discussion up to `until`: overlapping speakers merged, and pauses up to PAUSE_S
    between utterances bridged."""
    spans = sorted((s.t_start, min(s.t_end, until)) for s in segments if s.t_start < until)
    merged: list[tuple[float, float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= PAUSE_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def uncounted(
    segments: Sequence[TranscriptSegment], after: float | None, until: float
) -> list[tuple[float, float]]:
    """The stretches of discussion up to `until` that the tick which tracked up to `after` did
    not count.

    That tick counted up to where the speech it knew of ended, so a pause from there into this
    stretch counts now, like any other pause, and an utterance across `after` counts on both
    sides of it, once."""
    if after is None:
        counted_to = -math.inf
    else:
        known = [min(s.t_end, after) for s in segments if s.t_start < after]
        counted_to = min(after, max(known, default=after))
    return [(max(a, counted_to), b) for a, b in runs(segments, until) if b > counted_to]


def split_talk(
    spans: Sequence[tuple[float, float]],
    batch: Sequence[TranscriptSegment],
    about: Sequence[str | None],
) -> dict[str, float]:
    """Each item's seconds of `spans`. A segment's item holds from its start until the next
    segment of the batch (ordered by start) starts, so a pause goes with what was said before it
    and talk over it with the later speaker. The first segment's item also holds before it, and
    the last one's after it (an utterance still going at the tracked point). Time held by a
    segment about no item counts for none."""
    shares: dict[str, float] = {}
    for k, item in enumerate(about):
        lo = batch[k].t_start if k > 0 else -math.inf
        hi = batch[k + 1].t_start if k + 1 < len(batch) else math.inf
        held = sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in spans)
        if item is not None and held > 0:
            shares[item] = shares.get(item, 0.0) + held
    return shares


def scheduled_end_t(meeting: Meeting) -> float | None:
    """The scheduled end in seconds from the actual start; None without a duration. A meeting
    that started after its slot ran out (or within END_WARN_MIN of its end) gets its full
    duration from the actual start instead."""
    if meeting.duration_min is None or meeting.started_at is None:
        return None
    duration = meeting.duration_min * 60
    planned_start = meeting.scheduled_start or meeting.started_at
    end = (planned_start - meeting.started_at).total_seconds() + duration
    return end if end > END_WARN_MIN * 60 else duration


def seconds_since_start(meeting: Meeting) -> float:
    started = meeting.started_at or datetime.now(UTC)
    return max(0.0, (datetime.now(UTC) - started).total_seconds())


def not_come_up(item: AgendaItem, agenda: Agenda) -> bool:
    return item.status == "pending" and item.discussed_s == 0 and item.id != agenda.current_item_id


def nudge(agenda: Agenda, meeting: Meeting, now: float) -> tuple[Agenda, list[AgendaNudge]]:
    """At most one nudge per item, ever: about the next timeboxed item that has not come up when
    the current item has run past its timebox, and about every item that has not come up when
    the scheduled end is END_WARN_MIN or less away."""
    due: dict[str, str] = {}
    current = next((i for i in agenda.items if i.id == agenda.current_item_id), None)
    if current and current.minutes and current.discussed_s > current.minutes * 60:
        waiting = next((i for i in agenda.items if i.minutes and not_come_up(i, agenda)), None)
        if waiting and waiting.nudged_t is None:
            due[waiting.id] = (
                f"{waiting.title} hasn't come up yet; "
                f"{current.title} is past its {current.minutes} min."
            )

    end = scheduled_end_t(meeting)
    if end is not None and end - now <= END_WARN_MIN * 60:
        left = math.ceil((end - now) / 60)
        when = f"{left} min left" if left > 0 else "the scheduled end has passed"
        for item in agenda.items:
            if not_come_up(item, agenda) and item.nudged_t is None and item.id not in due:
                due[item.id] = f"{item.title} hasn't come up yet; {when}."

    if not due:
        return agenda, []
    items = [i.model_copy(update={"nudged_t": now}) if i.id in due else i for i in agenda.items]
    nudges = [
        AgendaNudge(meeting_id=meeting.id, item_id=i.id, text=due[i.id])
        for i in agenda.items
        if i.id in due
    ]
    return agenda.model_copy(update={"items": items}), nudges


@dataclass(frozen=True)
class Stretch:
    """What a tick learned: the point it tracked up to and, if the model was asked, the item
    being discussed at its end, each item's talk time in it and the items it finished, each with
    when its discussion ended."""

    until: float
    asked: bool = False
    current: str | None = None
    covered: Mapping[str, float] = field(default_factory=dict)
    seconds: Mapping[str, float] = field(default_factory=dict)


def advance(agenda: Agenda, stretch: Stretch) -> Agenda:
    """The agenda after a stretch: each item's talk time added, covered items marked. Ids no
    longer on the agenda (a person removed the item meanwhile) are ignored."""
    ids = {i.id for i in agenda.items}
    current = stretch.current if stretch.current in ids else None
    items = []
    for item in agenda.items:
        update: dict = {}
        if seconds := stretch.seconds.get(item.id, 0.0):
            update["discussed_s"] = item.discussed_s + seconds
        if item.id in stretch.covered and item.status == "pending":
            update |= {
                "status": "covered",
                "covered_by": AGENT_PARTICIPANT_ID,
                "covered_t": stretch.covered[item.id],
            }
        items.append(item.model_copy(update=update) if update else item)
    changes: dict = {"tracked_until": stretch.until, "items": items}
    if stretch.asked:
        changes["current_item_id"] = current
    return agenda.model_copy(update=changes)


def left_behind(agenda: Agenda, said: Labelled, seconds: Mapping[str, float]) -> frozenset[str]:
    """Pending items the team discussed and has left for another agenda item, whatever the model
    said of them: the stretch ends on a different item, and the item was under way (the one
    being discussed before the stretch, or one the stretch is partly about) with MOVED_ON_MIN_S
    of talk in all. Drifting into talk about no item leaves nothing behind, and an item that was
    not under way (one a person reopened earlier) is left as it is."""
    if said.current is None:
        return frozenset()
    under_way = {agenda.current_item_id, *said.about} - {None, said.current}
    return frozenset(
        item.id
        for item in agenda.items
        if item.id in under_way
        and item.status == "pending"
        and item.discussed_s + seconds.get(item.id, 0.0) >= MOVED_ON_MIN_S
    )


def ended_at(
    item_id: str, batch: Sequence[TranscriptSegment], about: Sequence[str | None]
) -> float:
    """When an item's discussion ended, for an item covered in this stretch: the end of the last
    segment about it, or, when nothing in the stretch is about it, the start of the stretch's
    first segment, which is when the talk had moved on."""
    own = [s.t_end for s, item in zip(batch, about, strict=True) if item == item_id]
    return max(own) if own else batch[0].t_start


class ClassificationFailed(Exception):
    """The model call failed or Gemini is not configured. Nothing was tracked, so the next tick
    retries the stretch; `response` still carries the rule nudges, which were saved."""

    def __init__(self, response: AgendaTrackResponse, error: LLMError):
        super().__init__(str(error))
        self.response = response
        self.error = error


def empty_agenda(meeting_id: str) -> Agenda:
    return Agenda(meeting_id=meeting_id, items=[], generated_at=datetime.now(UTC))


async def track_agenda(
    store: Store, make_llm: Callable[[], LLM], meeting: Meeting, now: float
) -> AgendaTrackResponse:
    """One tick at `now` seconds from the meeting start. The caller serialises ticks per meeting.

    The model is made and asked only when the stretch since the tracked point holds MIN_TALK_S
    of talk or has run MAX_WAIT_S. An item is covered when the model says its discussion is over
    or when the talk has left it for another item (left_behind), as of when that discussion
    ended, not of this tick. The result is saved with a compare-and-set on the agenda's
    revision; when someone else saved first (a lobby edit, another replica's tick), it is applied
    again to what they saved, without asking the model again. Raises ClassificationFailed when
    the model fails, and Conflict when the agenda kept changing under SAVE_ATTEMPTS saves."""
    agenda = await store.agenda(meeting.id)
    if agenda is None or not agenda.items:
        return AgendaTrackResponse(agenda=agenda or empty_agenda(meeting.id), nudges=[])

    since = agenda.tracked_until
    until = max(now - SETTLE_S, since or 0.0)
    transcript = await store.transcript(meeting.id)
    spans = uncounted(transcript, since, until)
    talk = sum(b - a for a, b in spans)
    batch = sorted(
        (s for s in transcript if (since is None or s.t_end > since) and s.t_end <= until),
        key=lambda s: s.t_start,
    )[-MAX_BATCH:]
    stretch: Stretch | None = None
    failure: LLMError | None = None
    if talk == 0:
        stretch = Stretch(until)  # nothing said since: move on without the model
    elif batch and (talk >= MIN_TALK_S or until - (since or 0.0) >= MAX_WAIT_S):
        earlier = sorted(
            (s for s in transcript if since is not None and s.t_end <= since),
            key=lambda s: s.t_start,
        )[-CONTEXT_SEGMENTS:]
        try:
            said = await classify(make_llm(), agenda, batch, earlier)
        except LLMError as e:
            failure = e
        else:
            seconds = split_talk(spans, batch, said.about)
            done = said.covered | left_behind(agenda, said, seconds)
            covered = {item: ended_at(item, batch, said.about) for item in done}
            stretch = Stretch(until, True, said.current, covered, seconds)

    response = await save(store, meeting, now, since, stretch)
    if failure is not None:
        raise ClassificationFailed(response, failure) from failure
    return response


async def save(
    store: Store, meeting: Meeting, now: float, since: float | None, stretch: Stretch | None
) -> AgendaTrackResponse:
    """Applies the stretch (if any) and the nudge rules to the latest agenda and saves it if that
    changed anything."""
    for _ in range(SAVE_ATTEMPTS):
        latest = await store.agenda(meeting.id)
        if latest is None or latest.tracked_until != since:  # another tick tracked it first
            return AgendaTrackResponse(agenda=latest or empty_agenda(meeting.id), nudges=[])
        updated, nudges = nudge(advance(latest, stretch) if stretch else latest, meeting, now)
        if updated == latest:
            return AgendaTrackResponse(agenda=latest, nudges=[])
        try:
            return AgendaTrackResponse(agenda=await store.save_agenda_if(updated), nudges=nudges)
        except Conflict:
            continue  # saved meanwhile: apply the same stretch to what was saved
    raise Conflict(f"the agenda of meeting {meeting.id} kept changing")
