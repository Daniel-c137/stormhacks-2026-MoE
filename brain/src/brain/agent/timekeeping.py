"""Keeping time against the agenda during a live meeting.

The realtime worker calls this on a timer. A tick takes the captions that became final since the
last one (often one or two utterances) and labels them with the items they are about in one model
call; it never runs per utterance as words arrive, and a tick with nothing new asks nothing and
saves nothing. Each item gets the talk time of its own segments, and items the team finished are
marked covered.

The tracked point is the end of the last caption labelled, not the time of the tick, so an
utterance still being spoken at a tick is counted whole once its caption arrives.

An item is covered when the model says its discussion is over, or when the team talked about it
and has since moved to another agenda item for a while (moved_on), as of the last thing said
about it. The tracker only covers an item that has come up, and one a person reopened only once
it comes up again. Nudges are decided by fixed rules, not the model, and are only shown: the
agent never speaks on its own.

A stretch of only small talk never covers an item. The model is Gemini, or Jev when JEV_MODEL is
set (agenda_jev): fast and cheap enough to be asked as soon as anything new has settled, so with
it the worker checks after every caption and a tick asks about a single short line too.
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

from .agenda_jev import Jev, jev_labels
from .ask import DATA_RULE, clip, fenced

SETTLE_S = 5.0  # captions ending this close to `now` wait a tick, so late finals are not skipped
# With Jev the worker checks 2.5 s after each caption ends (its AGENDA_CHECK_DELAY_SECONDS), so a
# caption has settled by then, and another speaker's finishing at about the same time has arrived.
JEV_SETTLE_S = 2.0
PAUSE_S = 15.0  # a pause up to this long between utterances still counts as discussion
# With the worker's tick every 10 s, the sentence that finishes an item is asked about at the
# first tick after it settles: within about 15 s, plus the model call. The worker's tick is what
# limits the model calls: at most one a tick, and none while nothing new has become final.
MIN_TALK_S = 4.0  # the model is asked once the new captions hold this much talk (a sentence),
MAX_WAIT_S = 10.0  # or once the oldest of them has waited this long (a one-word "done")
CONTEXT_SEGMENTS = 12  # lines before the stretch the model sees, to judge by but not to label
CONTEXT_MAX_AGE_S = 180.0  # and only those said this recently before it
MOVED_ON_MIN_S = 20.0  # talk an item needs before the team moving on covers it,
MOVED_ON_HOLD_S = 15.0  # and how long since its last word, with the talk on another item
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
        description="Labels of pending items whose discussion is over by the end of this "
        "stretch: decided, answered, closed, or really discussed and then left for another "
        "agenda item. Not an item the team is still weighing, nor one only named in passing.",
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
- A stretch is short, often one to three lines. The lines under "Earlier" came just before it:
  do not label them, but read them to see what a short remark ("yes, agreed") is about and where
  the talk came from.
- "covered" lists the pending items whose discussion is over by the end of this stretch:
  - the team decided or answered it. One clear closing sentence is enough, even as the last line
    of the stretch ("so the release moves to Monday");
  - or someone says it is done or closes it;
  - or they really discussed it (see how long it has been discussed, and the earlier lines) and
    the talk has since moved to another agenda item.
- Do not list an item the team is still weighing, one where only a side point was settled, one
  only named in passing ("we'll get to the launch date later"), or one nobody has discussed.
  Small talk after an item is not moving on.
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
    llm: LLM | None,
    agenda: Agenda,
    segments: Sequence[TranscriptSegment],
    earlier: Sequence[TranscriptSegment] = (),
    *,
    jev: Jev | None = None,
) -> Labelled:
    """Labels each segment (numbered from 1 in the order given) with an item; `earlier` is shown
    for context and not labelled. Labels not on the agenda and segment numbers outside the
    stretch are ignored; where runs overlap, the later one wins. Jev, when given, answers instead
    of the LLM."""
    labels = {f"a{n}": item for n, item in enumerate(agenda.items, 1)}
    if jev is not None:
        current = next((lb for lb, i in labels.items() if i.id == agenda.current_item_id), NO_ITEM)
        about, over = await jev_labels(jev, labels, current, segments, earlier)
        draft = AgendaTrackDraft(
            topics=[TopicRun(first=n, last=n, item=label) for n, label in enumerate(about, 1)],
            covered=sorted(over),
        )
    else:
        assert llm is not None
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
    before: str | None = None,
) -> dict[str, float]:
    """Each item's seconds of `spans`. A segment's item holds from its start until the next
    segment of the batch (ordered by start) starts, so a pause goes with what was said before it
    and talk over it with the later speaker. Time before the first segment starts (a pause since
    the tracked point) goes to `before`, the item being discussed until then, so a pause counts
    the same however the ticks fall. The last segment's item also holds after it (an utterance
    still going at the tracked point). Time held by a segment about no item counts for none."""
    shares: dict[str, float] = {}

    def add(item: str | None, lo: float, hi: float) -> None:
        held = sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in spans)
        if item is not None and held > 0:
            shares[item] = shares.get(item, 0.0) + held

    if batch:
        add(before, -math.inf, batch[0].t_start)
    for k, item in enumerate(about):
        add(item, batch[k].t_start, batch[k + 1].t_start if k + 1 < len(batch) else math.inf)
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
    """What an asked tick learned from its captions: the point it tracked up to (the end of the
    last of them), the item being discussed at their end, each item's talk time in them and the
    end of the last thing said about it, the items the model called covered, and the last agenda
    item the captions are about (small talk aside)."""

    until: float
    current: str | None = None
    seconds: Mapping[str, float] = field(default_factory=dict)
    last: Mapping[str, float] = field(default_factory=dict)
    said_covered: frozenset[str] = frozenset()
    ends_on: str | None = None


def moved_on(item: AgendaItem, stretch: Stretch) -> bool:
    """Whether the team has left this item for another: it has MOVED_ON_MIN_S of talk, the
    captions' last agenda item is a different one, and nothing has been said about it for
    MOVED_ON_HOLD_S. One line about something else, or a drift into small talk, is not that."""
    return (
        stretch.ends_on is not None
        and stretch.ends_on != item.id
        and item.last_discussed_t is not None
        and item.discussed_s >= MOVED_ON_MIN_S
        and stretch.until - item.last_discussed_t >= MOVED_ON_HOLD_S
    )


def advance(agenda: Agenda, stretch: Stretch) -> Agenda:
    """The agenda after a stretch: each item's talk time and last word added, covered items
    marked as of their last word. Ids no longer on the agenda (a person removed the item
    meanwhile) are ignored.

    What is covered is decided here, against the agenda as it is saved now: only a pending item
    that has come up (so not one a person reopened since, nor one nobody discussed), when the
    model called it covered or the team has moved on from it."""
    ids = {i.id for i in agenda.items}
    items = []
    for item in agenda.items:
        update: dict = {}
        if seconds := stretch.seconds.get(item.id, 0.0):
            update["discussed_s"] = item.discussed_s + seconds
        if item.id in stretch.last:
            update["last_discussed_t"] = max(stretch.last[item.id], item.last_discussed_t or 0.0)
        now = item.model_copy(update=update) if update else item
        if (
            now.status == "pending"
            and now.last_discussed_t is not None
            and (now.id in stretch.said_covered or moved_on(now, stretch))
        ):
            now = now.model_copy(
                update={
                    "status": "covered",
                    "covered_by": AGENT_PARTICIPANT_ID,
                    "covered_t": now.last_discussed_t,
                }
            )
        items.append(now)
    return agenda.model_copy(
        update={
            "tracked_until": stretch.until,
            "items": items,
            "current_item_id": stretch.current if stretch.current in ids else None,
        }
    )


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
    store: Store,
    make_llm: Callable[[], LLM],
    meeting: Meeting,
    now: float,
    *,
    jev: Jev | None = None,
) -> AgendaTrackResponse:
    """One tick at `now` seconds from the meeting start. The caller serialises ticks per meeting.

    The captions to track are the final segments that ended after the tracked point and have
    settled. The model is made and asked only when they hold MIN_TALK_S of talk or the oldest
    has waited MAX_WAIT_S; the tracked point then moves to the end of the last of them. With
    Jev, captions settle after JEV_SETTLE_S and any new one is asked about. A tick with nothing to
    ask about tracks nothing and saves nothing (but for a nudge that is due).
    The result is saved with a compare-and-set on the agenda's revision; when someone else saved
    first (a lobby edit, another replica's tick), it is applied again to what they saved,
    without asking the model again. Raises ClassificationFailed when the model fails, and
    Conflict when the agenda kept changing under SAVE_ATTEMPTS saves."""
    agenda = await store.agenda(meeting.id)
    if agenda is None or not agenda.items:
        return AgendaTrackResponse(agenda=agenda or empty_agenda(meeting.id), nudges=[])

    since = agenda.tracked_until
    settled = now - (SETTLE_S if jev is None else JEV_SETTLE_S)
    transcript = await store.transcript(meeting.id)
    batch = sorted(
        (s for s in transcript if (since is None or s.t_end > since) and s.t_end <= settled),
        key=lambda s: s.t_start,
    )[-MAX_BATCH:]
    stretch: Stretch | None = None
    failure: LLMError | None = None
    if batch:
        until = max(s.t_end for s in batch)
        spans = uncounted(transcript, since, until)
        talk = sum(b - a for a, b in spans)
        waited = settled - min(s.t_end for s in batch)
        if jev is not None or talk >= MIN_TALK_S or waited >= MAX_WAIT_S:
            first = batch[0].t_start
            earlier = sorted(
                (
                    s
                    for s in transcript
                    if since is not None
                    and s.t_end <= since
                    and s.t_end >= first - CONTEXT_MAX_AGE_S
                ),
                key=lambda s: s.t_start,
            )[-CONTEXT_SEGMENTS:]
            try:
                llm = make_llm() if jev is None else None
                said = await classify(llm, agenda, batch, earlier, jev=jev)
            except LLMError as e:
                failure = e
            else:
                last: dict[str, float] = {}
                for segment, item in zip(batch, said.about, strict=True):
                    if item is not None:
                        last[item] = max(segment.t_end, last.get(item, 0.0))
                stretch = Stretch(
                    until=until,
                    current=said.current,
                    seconds=split_talk(spans, batch, said.about, agenda.current_item_id),
                    last=last,
                    # small talk alone never finishes an item, whatever the model says
                    said_covered=(
                        said.covered if any(i is not None for i in said.about) else frozenset()
                    ),
                    ends_on=next((i for i in reversed(said.about) if i is not None), None),
                )

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
