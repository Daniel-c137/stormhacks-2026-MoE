"""Keeping time against the agenda during a live meeting.

The realtime worker calls this on a timer, never per utterance. Each tick classifies the final
segments since the last tracked point in one model call, gives their talk time to the item being
discussed and marks items the team finished. Nudges are decided by fixed rules, not the model,
and are only shown: the agent never speaks on its own.
"""

import math
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from brain.llm import LLM
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
END_WARN_MIN = 5  # nudge about items that have not come up this close to the scheduled end
MAX_BATCH = 200  # segments per classification; a longer backlog keeps the latest


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


def talk_time(segments: Sequence[TranscriptSegment], after: float) -> float:
    """Seconds of discussion after `after`: overlapping speakers count once, and pauses up to
    PAUSE_S between utterances count too."""
    spans = sorted((max(s.t_start, after), s.t_end) for s in segments if s.t_end > after)
    total = 0.0
    start = end = None
    for a, b in spans:
        if end is not None and a - end <= PAUSE_S:
            end = max(end, b)
            continue
        if end is not None:
            total += end - start
        start, end = a, b
    if end is not None:
        total += end - start
    return total


def scheduled_end_t(meeting: Meeting) -> float | None:
    """The scheduled end in seconds from the actual start; None without a duration."""
    if meeting.duration_min is None or meeting.started_at is None:
        return None
    planned_start = meeting.scheduled_start or meeting.started_at
    return (planned_start - meeting.started_at).total_seconds() + meeting.duration_min * 60


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


def advance(
    agenda: Agenda,
    tracked_until: float,
    classified: tuple[str | None, set[str]] | None,
    seconds: float,
) -> Agenda:
    """The agenda after a stretch: its time to the current item, covered items marked. Ids no
    longer on the agenda (a person removed the item meanwhile) are ignored."""
    changes: dict = {"tracked_until": tracked_until}
    if classified is not None:
        current, covered = classified
        ids = {i.id for i in agenda.items}
        current = current if current in ids else None
        items = []
        for item in agenda.items:
            update: dict = {}
            if item.id == current:
                update["discussed_s"] = item.discussed_s + seconds
            if item.id in covered and item.status == "pending":
                update["status"] = "covered"
            items.append(item.model_copy(update=update) if update else item)
        changes |= {"items": items, "current_item_id": current}
    return agenda.model_copy(update=changes)


async def track_agenda(store: Store, llm: LLM, meeting: Meeting, now: float) -> AgendaTrackResponse:
    """One tick at `now` seconds from the meeting start. The caller serialises ticks per meeting;
    across replicas the compare-and-set on tracked_until keeps a stretch from counting twice.
    An LLMError propagates with nothing saved, so the next tick retries the same stretch."""
    agenda = await store.agenda(meeting.id)
    if agenda is None or not agenda.items:
        empty = Agenda(meeting_id=meeting.id, items=[], generated_at=datetime.now(UTC))
        return AgendaTrackResponse(agenda=agenda or empty, nudges=[])

    since = agenda.tracked_until
    after = since if since is not None else -math.inf
    until = max(now - SETTLE_S, after, 0.0)
    batch = [s for s in await store.transcript(meeting.id) if after < s.t_end <= until]
    batch = batch[-MAX_BATCH:]
    classified, seconds = None, 0.0
    if batch:
        classified = await classify(llm, agenda, batch)
        seconds = talk_time(batch, after)

    # Edits may have landed while the model was thinking; apply the stretch to the latest list.
    latest = await store.agenda(meeting.id)
    if latest is None or latest.tracked_until != since:
        return AgendaTrackResponse(agenda=latest or agenda, nudges=[])
    updated, nudges = nudge(advance(latest, until, classified, seconds), meeting, now)
    if updated != latest:
        try:
            updated = await store.save_agenda_if(updated, tracked_until=since)
        except Conflict:  # another replica tracked this stretch first
            return AgendaTrackResponse(agenda=await store.agenda(meeting.id) or latest, nudges=[])
    return AgendaTrackResponse(agenda=updated, nudges=nudges)
