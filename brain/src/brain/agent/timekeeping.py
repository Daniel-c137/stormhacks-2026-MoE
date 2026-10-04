"""Keeping time against the agenda during a live meeting.

The realtime worker calls this on a timer, never per utterance. Each tick classifies the final
segments since the last tracked point in one model call, gives their talk time to the item being
discussed and marks items the team finished. Nudges are decided by fixed rules, not the model,
and are only shown: the agent never speaks on its own.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from brain.llm import LLM, LLMError
from brain.store import Conflict, Store
from contracts import (
    Agenda,
    AgendaItem,
    AgendaNudge,
    AgendaTrackResponse,
    Meeting,
    TranscriptSegment,
    get_identity,
)

SETTLE_S = 5.0  # segments ending this close to `now` wait a tick, so late finals are not skipped
PAUSE_S = 15.0  # a pause up to this long between utterances still counts as discussion
MIN_TALK_S = 20.0  # the model is asked once the new stretch holds this much talk,
MAX_WAIT_S = 60.0  # or once this long has passed since the tracked point
NOW_SLACK_S = 60.0  # how far a tick's `now` may run ahead of the brain's clock
END_WARN_MIN = 5  # nudge about items that have not come up this close to the scheduled end
MAX_BATCH = 200  # segments per classification; a longer backlog keeps the latest
SAVE_ATTEMPTS = 3  # saves of one tick's result before giving up on an agenda that keeps changing


class AgendaTrackDraft(BaseModel):
    current: str | None = Field(
        default=None,
        description="Label of the agenda item this stretch is mostly about, e.g. 'a2', or null "
        "when it is small talk, setup or a topic that is not on the agenda.",
    )
    covered: list[str] = Field(
        default=[],
        description="Labels of items the team clearly finished in this stretch: decided, "
        "answered or explicitly closed. Empty when unsure.",
    )


def track_system() -> str:
    agent = get_identity().agent_name
    return f"""You are {agent}. You keep time against a software team's meeting agenda. You read
the latest stretch of the live transcript and say which agenda item it is about.

Rules:
- "current" is the label of the item this stretch is mostly about, or null when it is not about
  any item on the agenda. Prefer the item the end of the stretch is about.
- "covered" lists items the team clearly finished in this stretch: they reached a decision or an
  answer, or said it is done and moved on. Mentioning an item does not cover it.
- Use only labels from the agenda. Judge only from the transcript."""


def clock(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"


def render_track_prompt(
    agenda: Agenda, labels: dict[str, AgendaItem], segments: Sequence[TranscriptSegment]
) -> str:
    def describe(label: str, item: AgendaItem) -> str:
        timebox = f", {item.minutes} min" if item.minutes else ""
        return f"[{label}] {item.title} ({item.status}{timebox})"

    current = next(
        (describe(lb, i) for lb, i in labels.items() if i.id == agenda.current_item_id), "none"
    )
    return "\n".join(
        [
            "Agenda ([label] title (status, timebox)):",
            *(describe(label, item) for label, item in labels.items()),
            "",
            f"Being discussed before this stretch: {current}",
            "",
            "Transcript stretch ([mm:ss] speaker: text):",
            *(f"[{clock(s.t_start)}] {s.speaker_name}: {s.text}" for s in segments),
        ]
    )


async def classify(
    llm: LLM, agenda: Agenda, segments: Sequence[TranscriptSegment]
) -> tuple[str | None, set[str]]:
    """(current item id or None, ids of covered items). Labels not on the agenda are dropped."""
    labels = {f"a{n}": item for n, item in enumerate(agenda.items, 1)}
    draft = await llm.generate_structured(
        render_track_prompt(agenda, labels, segments), AgendaTrackDraft, system=track_system()
    )

    def item_id(label: str | None) -> str | None:
        item = labels.get((label or "").strip())
        return item.id if item else None

    covered = {i for i in map(item_id, draft.covered) if i}
    return item_id(draft.current), covered


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


def talk_time(segments: Sequence[TranscriptSegment], after: float | None, until: float) -> float:
    """Seconds of discussion up to `until` that the tick which tracked up to `after` did not count.

    That tick counted up to where the speech it knew of ended, so a pause from there into this
    stretch counts now, like any other pause, and an utterance across `after` counts on both
    sides of it, once."""
    if after is None:
        counted_to = -math.inf
    else:
        known = [min(s.t_end, after) for s in segments if s.t_start < after]
        counted_to = min(after, max(known, default=after))
    return sum(max(0.0, b - max(a, counted_to)) for a, b in runs(segments, until))


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
    """What a tick learned: the point it tracked up to and, if the model was asked, which item
    the stretch was about (with its talk time) and which items it finished."""

    until: float
    asked: bool = False
    current: str | None = None
    covered: frozenset[str] = frozenset()
    seconds: float = 0.0


def advance(agenda: Agenda, stretch: Stretch) -> Agenda:
    """The agenda after a stretch: its time to the current item, covered items marked. Ids no
    longer on the agenda (a person removed the item meanwhile) are ignored."""
    ids = {i.id for i in agenda.items}
    current = stretch.current if stretch.current in ids else None
    items = []
    for item in agenda.items:
        update: dict = {}
        if item.id == current and stretch.seconds:
            update["discussed_s"] = item.discussed_s + stretch.seconds
        if item.id in stretch.covered and item.status == "pending":
            update["status"] = "covered"
        items.append(item.model_copy(update=update) if update else item)
    changes: dict = {"tracked_until": stretch.until, "items": items}
    if stretch.asked:
        changes["current_item_id"] = current
    return agenda.model_copy(update=changes)


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
    of talk or has run MAX_WAIT_S. The result is saved with a compare-and-set on the agenda's
    revision; when someone else saved first (a lobby edit, another replica's tick), it is applied
    again to what they saved, without asking the model again. Raises ClassificationFailed when
    the model fails, and Conflict when the agenda kept changing under SAVE_ATTEMPTS saves."""
    agenda = await store.agenda(meeting.id)
    if agenda is None or not agenda.items:
        return AgendaTrackResponse(agenda=agenda or empty_agenda(meeting.id), nudges=[])

    since = agenda.tracked_until
    until = max(now - SETTLE_S, since or 0.0)
    transcript = await store.transcript(meeting.id)
    talk = talk_time(transcript, since, until)
    batch = [s for s in transcript if (since is None or s.t_end > since) and s.t_end <= until]
    stretch: Stretch | None = None
    failure: LLMError | None = None
    if talk == 0:
        stretch = Stretch(until)  # nothing said since: move on without the model
    elif batch and (talk >= MIN_TALK_S or until - (since or 0.0) >= MAX_WAIT_S):
        try:
            current, covered = await classify(make_llm(), agenda, batch[-MAX_BATCH:])
        except LLMError as e:
            failure = e
        else:
            stretch = Stretch(until, True, current, frozenset(covered), talk)

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
