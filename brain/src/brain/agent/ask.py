"""Answering one deliberate question with two model calls.

1. Plan: the model picks up to MAX_TOOL_CALLS read-only tools, with arguments, from a fixed menu.
2. Execute: the tools run at the same time, each with a timeout, scoped to the asker's team. What
   they find is numbered as evidence, each item with its Source; a tool that is unconfigured or
   fails goes into Answer.unavailable.
3. Answer: the model writes the answer and names the evidence it relies on. Only evidence that
   exists is cited, and inference is marked as such. With no evidence the answer says so.

Nothing here stores, indexes or logs a question or an answer.
"""

import logging
import re
from collections.abc import Sequence
from datetime import date
from typing import Any, Literal
from uuid import uuid4

import anyio
from pydantic import BaseModel, Field

from brain.config import Settings
from brain.llm import LLM
from brain.memory import MeetingMemory
from brain.report.extraction import clock, speaker
from brain.store import Store
from contracts import (
    Answer,
    AskTurn,
    Invocation,
    Meeting,
    Person,
    Source,
    TranscriptSegment,
    get_identity,
)
from contracts.agent import Visibility

from .team_tools import (
    DEFAULT_TIMEOUT,
    Finding,
    TeamToolbox,
    github_reader,
    jira_reader,
    meeting_source,
)
from .tools import ToolSpec

log = logging.getLogger(__name__)

MAX_TOOL_CALLS = 4
MAX_EVIDENCE = 40
MAX_HISTORY_TURNS = 10
MAX_RECENT_SEGMENTS = 20
MAX_TEXT = 600

NO_EVIDENCE = "I couldn't find anything in the team's records that answers this, so I won't guess."
UNVERIFIED = "I couldn't verify an answer against any source, so I won't guess."


class Question(BaseModel):
    """One question to the agent, in a meeting (meeting_id set) or on Home."""

    id: str
    team_id: str
    text: str
    asker_id: str
    asker_name: str
    visibility: Visibility
    meeting_id: str | None = None
    history: list[AskTurn] = []  # oldest first
    recent: list[TranscriptSegment] = []  # final segments of this meeting, oldest first


class PlannedCall(BaseModel):
    tool: str = Field(description="A tool name from the menu, exactly as listed.")
    query: str | None = Field(
        default=None,
        description="search_meetings, decisions, jira_search, github_search: short search text.",
    )
    owner_id: str | None = Field(
        default=None, description='tasks: a team member id, or "me" for the asker.'
    )
    status: Literal["open", "overdue", "all"] | None = Field(
        default=None, description="tasks: open (default), overdue or all."
    )
    key: str | None = Field(default=None, description="jira_issue: the issue key, e.g. DS-104.")
    number: int | None = Field(
        default=None, description="github_read: the issue or pull request number."
    )
    kind: Literal["issue", "pr"] | None = Field(
        default=None, description="github_search, github_read: issue or pr (pull request)."
    )

    def arguments(self) -> dict[str, Any]:
        return self.model_dump(exclude={"tool"}, exclude_none=True)


class AskPlan(BaseModel):
    calls: list[PlannedCall] = Field(
        default=[], description="The lookups to run, most useful first. Empty if none is needed."
    )


class DraftAnswer(BaseModel):
    text: str = Field(description="The answer, using only facts from the evidence.")
    evidence_ids: list[str] = Field(
        default=[], description="Ids of every evidence item the answer relies on, e.g. ['e2']."
    )
    inference: str | None = Field(
        default=None,
        description="Anything concluded that no evidence states directly; otherwise null.",
    )


class Evidence(BaseModel):
    id: str
    finding: Finding


class ToolOrchestrator:
    """Answers deliberate questions with the team's read-only tools. Read-only: it never writes
    to the store, memory, Jira or GitHub."""

    def __init__(
        self,
        llm: LLM,
        store: Store,
        *,
        settings: Settings,
        memory: MeetingMemory | None = None,
        jira_target: Any = None,
        github_target: Any = None,
        max_calls: int = MAX_TOOL_CALLS,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.llm = llm
        self.store = store
        self.settings = settings
        self.memory = memory
        self.jira_target = jira_target
        self.github_target = github_target
        self.max_calls = max_calls
        self.timeout = timeout

    async def answer(
        self,
        invocation: Invocation,
        recent: list[TranscriptSegment],
        history: Sequence[AskTurn] = (),
    ) -> Answer:
        """The Orchestrator protocol: an in-meeting invocation, answered for the meeting's team."""
        meeting = await self.store.meeting(invocation.meeting_id)
        return await self.ask(
            Question(
                id=invocation.id,
                team_id=meeting.team_id,
                text=invocation.question,
                asker_id=invocation.asked_by_id,
                asker_name=invocation.asked_by_name,
                visibility=invocation.visibility,
                meeting_id=meeting.id,
                history=list(history),
                recent=recent,
            )
        )

    async def ask(self, question: Question) -> Answer:
        today = date.today()
        team_settings = await self.store.settings(question.team_id)
        members = await self.store.members(question.team_id)
        toolbox = TeamToolbox(
            question.team_id,
            question.asker_id,
            self.store,
            members=members,
            memory=self.memory,
            jira=jira_reader(self.settings, team_settings, self.jira_target),
            github=github_reader(self.settings, team_settings, self.github_target),
            timeout=self.timeout,
            today=today,
        )
        meeting = await toolbox.meeting(question.meeting_id) if question.meeting_id else None
        recent = recent_segments(question, meeting)
        context = render_context(question, meeting, members, today)

        plan = await self.llm.generate_structured(
            render_plan_prompt(context, question, recent, toolbox, self.max_calls),
            AskPlan,
            system=plan_system(self.max_calls),
        )
        findings, unavailable = await self.run(toolbox, plan.calls)
        if meeting is not None:
            findings = [
                Finding(
                    text=f"{speaker(s, get_identity().agent_name)}: {s.text}",
                    source=meeting_source(meeting, s.t_start),
                )
                for s in recent
            ] + findings
        evidence = number(findings)

        if not evidence:
            return Answer(
                id=str(uuid4()),
                invocation_id=question.id,
                text=NO_EVIDENCE,
                unavailable=unavailable,
            )
        draft = await self.llm.generate_structured(
            render_answer_prompt(context, question, evidence, unavailable),
            DraftAnswer,
            system=answer_system(question.visibility),
        )
        return finish(question, draft, evidence, unavailable)

    async def run(
        self, toolbox: TeamToolbox, calls: Sequence[PlannedCall]
    ) -> tuple[list[Finding], list[str]]:
        """Runs the planned calls at the same time; findings in plan order, then the reasons
        any source was unavailable."""
        menu = {spec.name for spec in toolbox.specs()}
        chosen: list[PlannedCall] = []
        for call in calls:
            if call.tool not in menu:
                log.warning("Refused tool %r: not one of the agent's read tools", call.tool)
            elif call not in chosen:
                chosen.append(call)
        chosen = chosen[: self.max_calls]

        results: list[Any] = [None] * len(chosen)

        async def run_one(i: int, call: PlannedCall) -> None:
            results[i] = await toolbox.call(call.tool, call.arguments())

        async with anyio.create_task_group() as group:
            for i, call in enumerate(chosen):
                group.start_soon(run_one, i, call)

        findings: list[Finding] = []
        unavailable: list[str] = []
        for result in results:
            if result.ok:
                findings += result.content
            elif result.error not in unavailable:
                unavailable.append(result.error)
        return findings, unavailable


def recent_segments(question: Question, meeting: Meeting | None) -> list[TranscriptSegment]:
    if meeting is None:
        return []
    final = [s for s in question.recent if s.is_final and s.meeting_id == meeting.id]
    return final[-MAX_RECENT_SEGMENTS:]


def number(findings: list[Finding]) -> list[Evidence]:
    """e1, e2, ... in order, without repeats."""
    seen: set[tuple[str, str]] = set()
    evidence: list[Evidence] = []
    for finding in findings:
        key = (finding.text, finding.source.model_dump_json())
        if key in seen:
            continue
        seen.add(key)
        evidence.append(Evidence(id=f"e{len(evidence) + 1}", finding=finding))
        if len(evidence) == MAX_EVIDENCE:
            break
    return evidence


MARKERS = re.compile(r"\s*\[\s*e\d+(?:\s*,\s*e\d+)*\s*\]", re.IGNORECASE)


def finish(
    question: Question, draft: DraftAnswer, evidence: list[Evidence], unavailable: list[str]
) -> Answer:
    """Keeps only cited evidence that exists, attaches its sources and marks inference."""
    by_id = {e.id: e for e in evidence}
    cited = [i.strip().strip("[]").lower() for i in draft.evidence_ids]
    valid = [i for i in dict.fromkeys(cited) if i in by_id]
    sources: list[Source] = []
    for i in valid:
        if by_id[i].finding.source not in sources:
            sources.append(by_id[i].finding.source)

    text = MARKERS.sub("", draft.text).strip()
    inference = MARKERS.sub("", draft.inference or "").strip()
    if cited and not valid:
        text, inference = UNVERIFIED, ""  # every claimed source is made up
    if not text:
        text = NO_EVIDENCE
    if inference:
        text += f"\n\nInference, not stated in any source: {inference}"
    return Answer(
        id=str(uuid4()),
        invocation_id=question.id,
        text=text,
        sources=sources,
        unavailable=unavailable,
    )


# prompts


def plan_system(max_calls: int) -> str:
    agent = get_identity().agent_name
    return f"""You are {agent}, the assistant of a software team. Someone asked you a question on \
purpose. Choose the read-only lookups that would find the facts to answer it.

Rules:
- Pick at most {max_calls} calls, using only tool names from the menu.
- Pick none when the conversation or the recent transcript already answers the question.
- "I", "me" and "my" mean the asker. For the asker's own tasks call tasks with owner_id "me".
- What the team said or decided lives in its meetings and decisions; the live state of issues
  and pull requests lives in Jira and GitHub.
- Search text is short: the key words of the topic, not the whole question.
- A tool marked not configured cannot return anything; pick it only when the question needs
  that source, so the answer can say it is unavailable.
- You only read. You never create, change or post anything."""


def answer_system(visibility: Visibility) -> str:
    agent = get_identity().agent_name
    audience = (
        "Only the asker sees this answer."
        if visibility == "private"
        else "Everyone in the conversation may see or hear this answer."
    )
    return f"""You are {agent}, the assistant of a software team, answering a question someone \
asked you on purpose. {audience}

Rules:
- Use only the numbered evidence and the conversation. Never add facts, names, dates, numbers or
  links that the evidence does not contain.
- Put the id of every evidence item the answer relies on in evidence_ids. Never write ids or
  brackets in the text.
- Say who said or decided something, and in which meeting, when the evidence shows it.
- Evidence can be off topic; ignore what does not answer the question.
- If the evidence does not answer the question, say so plainly instead of guessing.
- Anything you conclude that no evidence states directly goes in inference, not in text.
- Mention an unavailable source only when the question needed it.
- Plain text, no markdown, two to four short sentences that read well aloud."""


def render_context(
    question: Question, meeting: Meeting | None, members: list[Person], today: date
) -> list[str]:
    agent = get_identity().agent_name
    where = f'in the meeting "{meeting.title}"' if meeting else "on Home, outside any meeting"
    lines = [
        f"Today: {today.isoformat()} ({today:%A})",
        f"Asked by: {question.asker_name} (id {question.asker_id}), {where}",
        "",
        "Team members (id: name):",
        *(f"- {p.id}: {p.name}" for p in members),
    ]
    history = question.history[-MAX_HISTORY_TURNS:]
    if history:
        lines += ["", "Conversation so far (oldest first):"]
        lines += [
            f"{'Asker' if turn.role == 'user' else agent}: {clip(turn.text)}" for turn in history
        ]
    return lines


def render_plan_prompt(
    context: list[str],
    question: Question,
    recent: list[TranscriptSegment],
    toolbox: TeamToolbox,
    max_calls: int,
) -> str:
    agent = get_identity().agent_name
    lines = list(context)
    if recent:
        lines += ["", "Recent transcript of this meeting ([time] speaker: text):"]
        lines += [f"[{clock(s.t_start)}] {speaker(s, agent)}: {clip(s.text)}" for s in recent]
    lines += ["", f"Question: {question.text}", "", f"Tools (at most {max_calls} calls):"]
    lines += [render_tool(spec, toolbox.unavailable(spec.name)) for spec in toolbox.specs()]
    return "\n".join(lines)


def render_tool(spec: ToolSpec, unavailable: str | None) -> str:
    arguments = ", ".join(spec.parameters.get("properties", {}))
    line = f"- {spec.name}({arguments}): {spec.description}"
    return line + (f" [not configured: {unavailable}]" if unavailable else "")


def render_answer_prompt(
    context: list[str], question: Question, evidence: list[Evidence], unavailable: list[str]
) -> str:
    lines = [*context, "", f"Question: {question.text}", "", "Evidence ([id] source: content):"]
    for item in evidence:
        finding = item.finding
        when = f" ({finding.when.isoformat()})" if finding.when else ""
        lines.append(f"[{item.id}] {finding.source.label}{when}: {clip(finding.text)}")
    if unavailable:
        lines += ["", "Unavailable sources:", *(f"- {reason}" for reason in unavailable)]
    return "\n".join(lines)


def clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"
