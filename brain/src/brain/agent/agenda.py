"""The lobby agenda: rewrite a rough topic, and suggest items from the team's unfinished work.

Both are suggestions only; a person decides what goes on the agenda.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from brain.jira import JiraIssue
from brain.llm import LLM, LLMError
from brain.store import NotFound, Store
from brain.text import one_line
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
    minutes: int | None = Field(default=None, description="Suggested timebox, 5 to 30, or null.")
    source_ids: list[str] = Field(description="Ids of the inputs this is based on, e.g. ['i3'].")


class AgendaSuggestionDraft(BaseModel):
    items: list[SuggestedItem] = []


def meeting_label(meeting: Meeting) -> str:
    when = meeting.started_at or meeting.scheduled_start
    return f"{meeting.title} ({when:%Y-%m-%d})" if when else meeting.title


async def store_inputs(
    store: Store, team_id: str, meeting_id: str, today: date
) -> list[SuggestionInput]:
    """Open questions, blockers, risks and flagged decisions from the team's recent reports, then
    open task drafts and overdue tasks. Never the meeting being planned."""
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
        inputs += report_inputs(meeting, report)

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
            kind="meeting", label=meeting_label(meeting), meeting_id=meeting.id, t=task.t
        )
        inputs.append(SuggestionInput("Overdue task" if overdue else "Open task", text, source))
    return inputs


def report_inputs(meeting: Meeting, report: Report) -> list[SuggestionInput]:
    def source(t: float | None = None) -> Source:
        return Source(kind="meeting", label=meeting_label(meeting), meeting_id=meeting.id, t=t)

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
- Use only the inputs. Never add facts, names, dates, numbers or issue keys they do not contain.
- Every item lists the ids of the inputs it is based on. Merge inputs about the same topic.
- Titles are short and clear, like "Job queue: stay on Postgres or move to Redis".
- "why" is one plain sentence saying what is unfinished, from the inputs.
- Leave the list empty when nothing is worth discussing."""


def render_suggest_prompt(meeting: Meeting, today: date, labelled: dict[str, SuggestionInput]):
    lines = [f"[{label}] {i.kind} ({i.source.label}): {i.text}" for label, i in labelled.items()]
    return "\n".join(
        [
            f"Next meeting: {meeting.title}",
            f"Today: {today.isoformat()}",
            "",
            "Inputs ([id] kind (source): text):",
            *lines,
        ]
    )


async def suggest_items(
    llm: LLM, meeting: Meeting, inputs: Sequence[SuggestionInput], today: date
) -> list[AgendaItem]:
    """At most MAX_SUGGESTIONS proposed items. Each cites at least one input; any item that
    does not is dropped. Without inputs the model is not asked."""
    if not inputs:
        return []
    labelled = {f"i{n}": item for n, item in enumerate(inputs, 1)}
    draft = await llm.generate_structured(
        render_suggest_prompt(meeting, today, labelled),
        AgendaSuggestionDraft,
        system=suggest_system(),
    )
    items: list[AgendaItem] = []
    titles: set[str] = set()
    for suggested in draft.items:
        title = one_line(suggested.title)[:TITLE_MAX].rstrip()
        cited = [
            labelled[i]
            for i in dict.fromkeys(s.strip() for s in suggested.source_ids)
            if i in labelled
        ]
        if not title or not cited or title.casefold() in titles:
            continue
        titles.add(title.casefold())
        sources = list({tuple(s.source.model_dump().values()): s.source for s in cited}.values())
        items.append(
            AgendaItem(
                id=uuid4().hex,
                title=title,
                why=one_line(suggested.why) or f"{cited[0].kind}: {cited[0].text}",
                sources=sources,
                minutes=suggested.minutes if valid_minutes(suggested.minutes) else None,
            )
        )
        if len(items) == MAX_SUGGESTIONS:
            break
    return items
