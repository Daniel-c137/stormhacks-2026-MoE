"""What Gemini is asked for. The builder checks every answer against the transcript."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from contracts import AGENT_PARTICIPANT_ID, AgendaItem, Person, TranscriptSegment, get_identity
from contracts.agent import SourceKind

EVIDENCE = "Ids of the transcript segments that support this, e.g. ['s4']."


class ExtractedTask(BaseModel):
    title: str = Field(description="Imperative and short, e.g. 'Refund the double-charged users'.")
    description: str | None = Field(
        default=None, description="Only what the transcript adds to the title; otherwise null."
    )
    owner_id: str | None = Field(
        default=None,
        description="Participant id of who will do it, only when stated or clearly implied; "
        "otherwise null.",
    )
    due: str | None = Field(
        default=None,
        description="YYYY-MM-DD, only when a deadline is stated; otherwise null.",
    )
    evidence: list[str] = Field(description=EVIDENCE)


class ExtractedDecision(BaseModel):
    text: str
    made_by_id: str | None = Field(default=None, description="Participant id, if clear.")
    evidence: list[str] = Field(description=EVIDENCE)


class ExtractedRisk(BaseModel):
    text: str
    severity: Literal["low", "high"]
    evidence: list[str] = Field(description=EVIDENCE)


class ExtractedLink(BaseModel):
    kind: SourceKind
    label: str = Field(description="Exactly as said, e.g. 'DS-104', 'PR 212' or a URL.")
    url: str | None = None


class ReportExtraction(BaseModel):
    summary: str = Field(description="Two to five plain sentences.")
    topics: list[str] = []
    decisions: list[ExtractedDecision] = []
    tasks: list[ExtractedTask] = []
    risks: list[ExtractedRisk] = []
    open_questions: list[str] = []
    blockers: list[str] = []
    technical_context: list[str] = []
    links: list[ExtractedLink] = []


def system_prompt() -> str:
    agent = get_identity().agent_name
    return f"""You write the record of a software team's meeting from its transcript.

Rules:
- Use only the transcript. Never add facts, names, dates or links it does not contain.
- Every decision, task and risk lists the ids of the segments that support it.
- Tasks are concrete follow-up work someone agreed to do or was asked to do.
- A task's owner is a participant id, only when the transcript states or clearly implies who
  will do it ("I'll take it" means the speaker). Otherwise null. {agent} is the meeting
  assistant and is never an owner.
- A due date only when a deadline is stated, as YYYY-MM-DD. Resolve relative days such as
  "by Wednesday" against the meeting date. Otherwise null.
- Decisions are what the team agreed on, not proposals that are still open.
- Open questions are unresolved; blockers are things stopping work.
- Links are issue keys, pull request numbers or URLs mentioned in the transcript, as said.
- Leave a list empty when nothing applies. Never fill a section just to have something."""


def render_prompt(
    *,
    title: str,
    meeting_date: str | None,
    people: list[Person],
    labelled: dict[str, TranscriptSegment],
    agenda: Sequence[AgendaItem] = (),
) -> str:
    agent = get_identity().agent_name
    participants = [f"- {p.id}: {p.name}" for p in people]
    participants.append(f"- {AGENT_PARTICIPANT_ID}: {agent} (meeting assistant)")
    lines = [
        f"[{label} {clock(s.t_start)}] {speaker(s, agent)}: {s.text}"
        for label, s in labelled.items()
    ]
    return "\n".join(
        [
            f"Meeting: {title}",
            f"Date: {meeting_date or 'unknown'}",
            "",
            "Participants (id: name):",
            *participants,
            "",
            *agenda_lines(agenda),
            "Transcript ([segment time] speaker: text):",
            *lines,
        ]
    )


def agenda_lines(agenda: Sequence[AgendaItem]) -> list[str]:
    """The planned items in order, or nothing. Topics follow it only where the transcript does."""
    if not agenda:
        return []
    items = [
        f"{i}. {item.title}" + (f" ({item.minutes} min)" if item.minutes else "")
        for i, item in enumerate(agenda, 1)
    ]
    return [
        "Agenda (planned items, in order):",
        *items,
        "List the topics in agenda order, named after the agenda items the transcript discusses,",
        "then any other topics discussed. Leave out agenda items the transcript never discusses.",
        "",
    ]


def speaker(segment: TranscriptSegment, agent: str) -> str:
    return agent if segment.speaker_id == AGENT_PARTICIPANT_ID else segment.speaker_name


def clock(t: float) -> str:
    total = int(t)
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"
