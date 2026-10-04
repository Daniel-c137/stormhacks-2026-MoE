"""The lobby agenda: rewrite a rough topic, and suggest items from the team's unfinished work.

Both are suggestions only; a person decides what goes on the agenda.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, tzinfo
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from brain.jira import JiraIssue
from brain.llm import LLM, LLMError
from brain.store import NotFound, Store
from brain.text import one_line
from brain.zones import local_date
from contracts import (
    Agenda,
    AgendaItem,
    Meeting,
    Report,
    Source,
    get_identity,
)

TITLE_MAX = 200
TOPIC_MAX = 1000
MINUTES_MIN, MINUTES_MAX = 1, 240
MAX_SUGGESTIONS = 6
# Suggested timeboxes, in minutes; a meeting's length caps their total.
TIMEBOX_MIN, TIMEBOX_MAX, TIMEBOX_DEFAULT = 5, 30, 10
RECENT_REPORTS = 5
MAX_TASKS = 15


class AgendaPlanner(Protocol):
    async def build(self, meeting: Meeting) -> Agenda:
        """From previous summaries, open tasks and unfinished GitHub/Jira work."""
        ...


def valid_minutes(minutes: int | None) -> bool:
    return minutes is None or MINUTES_MIN <= minutes <= MINUTES_MAX


# rewrite


class AgendaRewrite(BaseModel):
    title: str = Field(
        description="One short agenda item, at most about ten words, e.g. "
        "'Job queue: stay on Postgres or move to Redis'."
    )


def rewrite_system() -> str:
    agent = get_identity().agent_name
    return f"""You are {agent}. A person typed a rough topic for a software team's meeting agenda.
Rewrite it as one clear, short agenda item.

Rules:
- Keep the person's meaning. Never add facts, names, numbers, systems, dates or decisions
  that the topic does not contain.
- At most about ten words, on one line, no trailing period.
- A rough question may become a choice to discuss, e.g. "tabs or spaces?" becomes
  "Code style: tabs or spaces"."""


async def rewrite_topic(llm: LLM, text: str) -> str:
    """One clear agenda item for a rough topic. Raises ValueError for a blank topic."""
    topic = text.strip()
    if not topic:
        raise ValueError("Type a topic to rewrite")
    out = await llm.generate_structured(f"Topic: {topic}", AgendaRewrite, system=rewrite_system())
    title = one_line(out.title)
    if not title:
        raise LLMError("The model returned an empty agenda item")
    return title[:TITLE_MAX].rstrip()


# suggestions


@dataclass(frozen=True)
class SuggestionInput:
    """One piece of unfinished work the model may build a suggestion on, with its source."""

    kind: str
    text: str
    source: Source


class SuggestedItem(BaseModel):
    title: str = Field(description="A short, clear agenda item, at most about ten words.")
    why: str = Field(description="One sentence on why it is worth discussing, from the inputs.")
    minutes: int | None = Field(
        default=None, description=f"Timebox in minutes, {TIMEBOX_MIN} to {TIMEBOX_MAX}."
    )
    source_ids: list[str] = Field(description="Ids of the inputs this is based on, e.g. ['i3'].")


class AgendaSuggestionDraft(BaseModel):
    items: list[SuggestedItem] = []


def title_key(title: str) -> str:
    """A title as compared for duplicates: without case, punctuation or extra spaces."""
    return " ".join(re.sub(r"[\W_]+", " ", title.casefold()).split())


TRAILING_NOTE = re.compile(r"\s*[(\[]([^()\[\]]*)[)\]]\s*$")


def without_source(title: str, cited: Sequence[SuggestionInput]) -> str:
    """The title without a trailing "(DS-117)" or "[Checkout sync]" naming one of its sources,
    which are attached to the item anyway."""
    note = TRAILING_NOTE.search(title)
    rest = title[: note.start()].rstrip() if note else ""
    if not note or not rest:
        return title
    named = title_key(note.group(1))
    labels = {title_key(i.source.label) for i in cited}
    labels |= {title_key(TRAILING_NOTE.sub("", i.source.label)) for i in cited}
    return rest if len(named) > 2 and any(named in label for label in labels if label) else title


def fit_timeboxes(wanted: Sequence[int], budget: int | None) -> list[int | None]:
    """Timeboxes whose total fits `budget` minutes (None: no limit). They shrink toward
    TIMEBOX_MIN in proportion; items that do not fit even at the minimum get none."""
    if budget is None or sum(wanted) <= budget:
        return list(wanted)
    head = list(wanted[: max(budget, 0) // TIMEBOX_MIN])
    total = sum(head)
    if total > budget:
        spare, extra = budget - TIMEBOX_MIN * len(head), total - TIMEBOX_MIN * len(head)
        head = [TIMEBOX_MIN + (m - TIMEBOX_MIN) * spare // extra for m in head]
    return [*head, *[None] * (len(wanted) - len(head))]


def time_left(meeting: Meeting, existing: Sequence[AgendaItem]) -> int | None:
    """Minutes of the meeting not yet given to an agenda item, or None without a length."""
    if meeting.duration_min is None:
        return None
    return max(meeting.duration_min - sum(i.minutes or 0 for i in existing), 0)


def meeting_label(meeting: Meeting, zone: tzinfo = UTC) -> str:
    """The meeting's title and its day in the team's zone."""
    day = local_date(meeting.started_at or meeting.scheduled_start, zone)
    return f"{meeting.title} ({day.isoformat()})" if day else meeting.title


async def store_inputs(
    store: Store, team_id: str, meeting_id: str, today: date, zone: tzinfo = UTC
) -> list[SuggestionInput]:
    """Open questions, blockers, risks and flagged decisions from the team's recent reports, then
    open task drafts and overdue tasks. Never the meeting being planned. `today` and the meetings'
    days are the team's, in `zone`."""
    meetings = {m.id: m for m in await store.meetings(team_id) if m.id != meeting_id}
    inputs: list[SuggestionInput] = []
    reports = 0
    for meeting in meetings.values():
        if reports == RECENT_REPORTS:
            break
        try:
            report = await store.report(meeting.id)
        except NotFound:
            continue
        reports += 1
        inputs += report_inputs(meeting, report, zone)

    tasks = 0
    for task in await store.tasks(team_id):
        meeting = meetings.get(task.meeting_id)
        if meeting is None or not task.include or task.jira_status == "done" or tasks == MAX_TASKS:
            continue
        overdue = task.due is not None and task.due < today
        if not overdue and (task.key or task.jira_status != "draft"):
            continue
        tasks += 1
        details = [f"due {task.due.isoformat()}"] if task.due else []
        if task.key:
            details.append(f"{task.key}, {task.jira_status}")
        text = task.title + (f" ({'; '.join(details)})" if details else "")
        source = Source(
            kind="meeting", label=meeting_label(meeting, zone), meeting_id=meeting.id, t=task.t
        )
        inputs.append(SuggestionInput("Overdue task" if overdue else "Open task", text, source))
    return inputs


def report_inputs(meeting: Meeting, report: Report, zone: tzinfo = UTC) -> list[SuggestionInput]:
    def source(t: float | None = None) -> Source:
        label = meeting_label(meeting, zone)
        return Source(kind="meeting", label=label, meeting_id=meeting.id, t=t)

    inputs = [SuggestionInput("Open question", q, source()) for q in report.open_questions]
    inputs += [SuggestionInput("Blocker", b, source()) for b in report.blockers]
    inputs += [SuggestionInput(f"Risk ({r.severity})", r.text, source()) for r in report.risks]
    for decision in report.decisions:
        if decision.status == "superseded":
            kind = "Superseded decision"
        elif decision.relation is not None and decision.relation.type == "contradicts":
            kind = "Decision contradicting an earlier one"
        else:
            continue
        inputs.append(SuggestionInput(kind, decision.text, source(decision.t)))
    return [i for i in inputs if i.text.strip()]


def jira_inputs(issues: Sequence[JiraIssue]) -> list[SuggestionInput]:
    inputs = []
    for issue in issues:
        details = ", ".join(d for d in (issue.status, issue.assignee) if d)
        text = f"{issue.key}: {issue.summary}" + (f" ({details})" if details else "")
        source = Source(kind="jira_issue", label=issue.key, url=issue.url)
        inputs.append(SuggestionInput("Unfinished Jira issue", text, source))
    return inputs


def suggest_system() -> str:
    agent = get_identity().agent_name
    return f"""You are {agent}. You propose agenda items for a software team's next meeting from
their unfinished work: earlier meetings' open questions, blockers, risks and reversed decisions,
open or overdue tasks, and unfinished Jira issues.

Rules:
- Propose at most {MAX_SUGGESTIONS} items, the ones most worth the team's time. Fewer is fine.
- Most important first.
- Never propose a topic that is already on the agenda, even in other words.
- Use only the inputs. Never add facts, names, dates, numbers or issue keys they do not contain.
- Every item lists the ids of the inputs it is based on. Merge inputs about the same topic.
- Titles are short and clear, like "Job queue: stay on Postgres or move to Redis". A title names
  the topic, not where it came from: no issue key, meeting name or date in the title, since the
  sources are attached separately, unless the title would be unclear without the issue key.
- "why" is one plain sentence saying what is unfinished, from the inputs.
- Give every item a timebox of {TIMEBOX_MIN} to {TIMEBOX_MAX} minutes, sized to the topic. When the
  time left is given, keep the total within it, proposing fewer items if needed.
- Leave the list empty when nothing is worth discussing."""


def render_suggest_prompt(
    meeting: Meeting,
    today: date,
    labelled: dict[str, SuggestionInput],
    existing: Sequence[AgendaItem] = (),
) -> str:
    lines = [f"[{label}] {i.kind} ({i.source.label}): {i.text}" for label, i in labelled.items()]
    head = [f"Next meeting: {meeting.title}", f"Today: {today.isoformat()}"]
    left = time_left(meeting, existing)
    if left is not None:
        head += [
            f"Meeting length: {meeting.duration_min} min",
            f"Time left for new items: {left} min",
        ]
    if existing:
        head += ["", "Already on the agenda:", *(f"- {i.title}" for i in existing)]
    return "\n".join([*head, "", "Inputs ([id] kind (source): text):", *lines])


async def suggest_items(
    llm: LLM,
    meeting: Meeting,
    inputs: Sequence[SuggestionInput],
    today: date,
    existing: Sequence[AgendaItem] = (),
) -> list[AgendaItem]:
    """At most MAX_SUGGESTIONS items, ready to add to the agenda after `existing`. Each cites at
    least one input; any item that does not, or repeats a title already on the agenda or earlier
    in the list (ignoring case and punctuation), is dropped. Every item gets a timebox, and their
    total fits the meeting's time left. Without inputs the model is not asked."""
    if not inputs:
        return []
    labelled = {f"i{n}": item for n, item in enumerate(inputs, 1)}
    draft = await llm.generate_structured(
        render_suggest_prompt(meeting, today, labelled, existing),
        AgendaSuggestionDraft,
        system=suggest_system(),
    )
    items: list[AgendaItem] = []
    titles = {title_key(i.title) for i in existing}
    for suggested in draft.items:
        cited = [
            labelled[i]
            for i in dict.fromkeys(s.strip() for s in suggested.source_ids)
            if i in labelled
        ]
        title = without_source(one_line(suggested.title), cited)[:TITLE_MAX].rstrip()
        if not title or not cited or title_key(title) in titles:
            continue
        titles.add(title_key(title))
        sources = list({tuple(s.source.model_dump().values()): s.source for s in cited}.values())
        minutes = suggested.minutes or TIMEBOX_DEFAULT
        items.append(
            AgendaItem(
                id=uuid4().hex,
                title=title,
                why=one_line(suggested.why) or f"{cited[0].kind}: {cited[0].text}",
                sources=sources,
                minutes=min(max(minutes, TIMEBOX_MIN), TIMEBOX_MAX),
            )
        )
        if len(items) == MAX_SUGGESTIONS:
            break
    timeboxes = fit_timeboxes([i.minutes or 0 for i in items], time_left(meeting, existing))
    return [i.model_copy(update={"minutes": m}) for i, m in zip(items, timeboxes, strict=True)]
