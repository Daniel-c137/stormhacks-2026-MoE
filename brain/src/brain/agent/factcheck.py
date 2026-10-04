"""Fact-checking technical claims during a live meeting.

The realtime worker calls this on a timer, never per utterance, and never while the team's
sensitivity is quiet. Each tick:

1. Reads the final segments since the last checked point and keeps those a cheap, deterministic
   filter finds checkable: pull request and issue numbers, Jira keys, versions, "merged",
   "released", "in the latest release", "we decided", deadlines, configuration values. Nothing
   passes: no model call.
2. Rate limit: one model check per MIN_INTERVAL_S at most, MAX_CLAIMS claims a batch at most.
3. Plan: one model call picks the claims worth checking and read-only lookups for each.
4. Evidence: the team's tools (the ones ask.py uses) and, when configured, code search; numbered
   like an answer's evidence.
5. Verdicts: one model call for the batch. Only evidence that exists counts, confidence is
   clamped to [0, 1], and a verdict without evidence is unknown.

Nothing goes to the room. Every check returned is addressed (recipient_id) to the person who
made the claim, and the worker sends it to them alone as a private chat message: each
contradiction, and at eager a claim that could not be verified. Nobody said anything wrong when a
claim is supported, so it is never sent. A confident, high-severity contradiction, and at eager a
supported claim, is kept for the write-up without its recipient; the rest is never stored,
indexed or logged, and the agent never speaks a check.
"""

import math
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import anyio
from pydantic import BaseModel, Field

from brain.config import Settings
from brain.github import GitHubReader
from brain.jira import root_cause
from brain.llm import LLM
from brain.memory import MeetingMemory
from brain.report.extraction import by_agent, clock
from brain.store import Conflict, FactCheckState, Store
from contracts import (
    CodeSnippet,
    FactCheck,
    FactCheckResponse,
    Meeting,
    Source,
    TranscriptSegment,
    get_identity,
)
from contracts.agent import Severity, Verdict
from contracts.settings import Sensitivity

from .ask import (
    DATA_RULE,
    Evidence,
    PlannedCall,
    ToolOrchestrator,
    clip,
    dated,
    fenced,
    number,
    render_tool,
)
from .team_tools import DEFAULT_TIMEOUT, Finding, TeamToolbox

SETTLE_S = 5.0  # segments ending this close to `now` wait a tick, so late finals are not skipped
MAX_BATCH = 200  # segments read per tick; a longer backlog keeps the latest
MIN_INTERVAL_S: dict[Sensitivity, float] = {"balanced": 120.0, "eager": 60.0}
MAX_CLAIMS: dict[Sensitivity, int] = {"balanced": 3, "eager": 5}
MAX_LOOKUPS = 6  # tool calls per batch, across its claims
MAX_EVIDENCE = 30
MAX_CODE_QUERIES = 2
MAX_SNIPPETS = 3  # per code query
KEEP_CONFIDENCE = 0.8  # a high-severity contradiction at least this sure is kept for the report
MAX_FINDING = 240

NO_CODE_SEARCH = "Code search is not available yet"
# Code is looked up through each claim's code_query, so the toolbox's own code tool stays off
# the fact-check menu and is never run from a plan.
OWN_CODE_TOOL = "github_code"


# the claim filter

_WHICH = r"(?:latest|last|current|new|newest|next) "
_STATUS = (
    r"merged|released|shipped|deployed|closed|fixed|resolved|reverted|landed|launched"
    r"|rolled (?:back|out)|went (?:out|live)|is live|is out|in prod(?:uction)?"
    # release membership: "in the latest release", "in v1.2.3", "made it into the release"
    rf"|in (?:the )?{_WHICH}release|in v\d+(?:\.\d+)*|part of (?:the )?(?:{_WHICH})?release"
    rf"|made (?:it into )?(?:the )?(?:{_WHICH})?release|shipped in"
)
_DECISION = (
    r"we (?:decided|agreed|chose|settled on|went with)|(?:decision|agreement) (?:was|is)"
    r"|we(?:'re| are) going with"
)
_DEADLINE = r"deadline|due|eta|target date|cut-?off"
_CONFIG = (
    r"timeout|limit|ttl|threshold|max(?:imum)?|min(?:imum)?|retries|retry|rate|pool size"
    r"|batch size|interval|quota|port|replicas|workers"
)
TOPIC = re.compile(rf"\b(?:{_STATUS}|{_DECISION}|{_DEADLINE}|{_CONFIG})\b", re.IGNORECASE)

_REFERENCE = r"\b(?:pr|pull request|issue|ticket|bug)\s*(?:number\s*)?#?\s*\d+\b|(?<!\w)#\d+\b"
_VERSION = r"\bv\d+(?:\.\d+)+\b|\b\d+\.\d+\.\d+\b|\bversion \d+(?:\.\d+)*\b"
_DATE = (
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tonight|tomorrow"
    r"|yesterday|(?:last|next|this) (?:week|month|sprint|quarter)"
    r"|end of (?:the )?(?:day|week|month|sprint|quarter)|eod|eow|q[1-4]"
    r"|january|february|april|june|july|august|september|october|november|december"
    r"|jan|feb|apr|jun|jul|aug|sept?|oct|nov|dec)\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}\b"
)
_QUANTITY = (
    r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?"
    r"|days?|%|percent|kb|mb|gb|tb|retries|requests?|rps|qps|connections?|workers?|replicas?"
    r"|threads?)(?!\w)"
)
REFERENT = re.compile(rf"{_REFERENCE}|{_VERSION}|{_DATE}|{_QUANTITY}", re.IGNORECASE)
JIRA_KEY = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d+\b")


def is_claim(text: str, sensitivity: Sensitivity) -> bool:
    """Whether a sentence is a technical, checkable claim worth a model's look. Balanced wants
    a status, decision, deadline or setting about something concrete (a number, key, version or
    date); eager takes either alone. Questions are never claims."""
    if sensitivity == "quiet" or text.rstrip().endswith("?"):
        return False
    topic = TOPIC.search(text) is not None
    referent = REFERENT.search(text) is not None or JIRA_KEY.search(text) is not None
    return topic and referent if sensitivity == "balanced" else topic or referent


def claim_candidates(
    segments: Iterable[TranscriptSegment], sensitivity: Sensitivity
) -> list[TranscriptSegment]:
    """People's final segments that pass the filter, in order. The agent's own words never do."""
    return [s for s in segments if s.is_final and not by_agent(s) and is_claim(s.text, sensitivity)]


# what the model is asked for


class ClaimPlan(BaseModel):
    claim: str = Field(description="The claim's label, e.g. 'c1'.")
    calls: list[PlannedCall] = Field(
        default=[], description="Read-only lookups that would settle this claim, most useful first."
    )
    code_query: str | None = Field(
        default=None,
        description="Short search text for the team's code, only when code search is available "
        "and the claim is about code or configuration; otherwise null.",
    )


class FactCheckPlan(BaseModel):
    claims: list[ClaimPlan] = Field(
        default=[], description="The claims worth checking, most important first. Empty if none."
    )


class ClaimVerdict(BaseModel):
    claim: str = Field(description="The claim's label, e.g. 'c1'.")
    verdict: Verdict = Field(
        description="contradicted when the evidence clearly shows the claim is wrong, supported "
        "when it clearly confirms it, otherwise unknown."
    )
    confidence: float = Field(description="0 to 1: how sure the evidence makes you.")
    severity: Severity = Field(
        description="high when acting on a wrong claim could cause real harm now; otherwise low."
    )
    evidence_ids: list[str] = Field(
        default=[], description="Ids of every evidence item the verdict relies on, e.g. ['e2']."
    )
    finding: str = Field(
        default="",
        description="One short, factual sentence on what the evidence shows about the claim, "
        "e.g. 'PR #41 was merged on 30 September, after the latest release (v0.9.3, 28 "
        "September)'. No evidence ids.",
    )


class FactCheckVerdicts(BaseModel):
    checks: list[ClaimVerdict] = []


# the checker

CodeLookup = Callable[[GitHubReader, str], Awaitable[list[CodeSnippet]]]
"""Code evidence for a search: the team's repository and short search text in, snippets out."""


class Settled(BaseModel):
    """A verdict after the code's checks."""

    verdict: Verdict
    confidence: float
    severity: Severity
    finding: str = ""
    sources: list[Source] = []
    snippets: list[CodeSnippet] = []


class FactChecker:
    """Checks a live meeting's claims with the team's read-only tools. It writes only its own
    progress and the checks worth keeping, for the report; never to memory, Jira or GitHub.

    `llm` is a model, or a factory called only once a claim needs checking, so a quiet team or a
    stretch of small talk never needs one. `code` adds code evidence when code search is set up.
    """

    def __init__(
        self,
        llm: LLM | Callable[[], LLM],
        store: Store,
        *,
        settings: Settings,
        memory: MeetingMemory | None = None,
        jira_target: Any = None,
        github_target: Any = None,
        code: CodeLookup | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._llm = llm
        self.store = store
        self.settings = settings
        self.memory = memory
        self.jira_target = jira_target
        self.github_target = github_target
        self.code = code
        self.timeout = timeout

    def model(self) -> LLM:
        return self._llm() if callable(self._llm) else self._llm

    async def tick(self, meeting: Meeting, now: float) -> FactCheckResponse:
        """One tick at `now` seconds from the meeting start. The caller serialises ticks per
        meeting; across replicas the compare-and-set on checked_until keeps a stretch from being
        checked twice. An LLMError propagates with nothing saved, so the next tick retries."""
        team = await self.store.settings(meeting.team_id)
        sensitivity = team.sensitivity
        if sensitivity == "quiet":
            return FactCheckResponse()

        state = await self.store.fact_check_state(meeting.id)
        state = state or FactCheckState(meeting_id=meeting.id)
        since = state.checked_until
        after = since if since is not None else -math.inf
        until = max(now - SETTLE_S, after, 0.0)
        batch = [s for s in await self.store.transcript(meeting.id) if after < s.t_end <= until]
        if not batch:
            return FactCheckResponse()
        claims = claim_candidates(batch[-MAX_BATCH:], sensitivity)
        if not claims:
            await self._save(state.model_copy(update={"checked_until": until}), since)
            return FactCheckResponse()
        if state.checked_at is not None and now - state.checked_at < MIN_INTERVAL_S[sensitivity]:
            return FactCheckResponse()  # these claims wait for the next tick

        labelled = {f"c{n}": s for n, s in enumerate(claims[-MAX_CLAIMS[sensitivity] :], 1)}
        settled = await self._check(meeting, labelled, sensitivity)

        checks: list[FactCheck] = []  # each sent only to whoever made the claim
        kept: list[FactCheck] = []  # for the write-up
        for label, segment in labelled.items():
            if label not in settled:
                continue
            found = settled[label]
            check = fact_check(segment, found)
            if found.verdict == "contradicted" or (
                found.verdict == "unknown" and sensitivity == "eager"
            ):
                checks.append(check)
            if (
                found.verdict == "contradicted"
                and found.severity == "high"
                and found.confidence >= KEEP_CONFIDENCE
            ) or (found.verdict == "supported" and sensitivity == "eager"):
                kept.append(check.model_copy(update={"recipient_id": None}))

        changes = {"checked_until": until, "checked_at": now}
        if not await self._save(state.model_copy(update=changes), since):
            return FactCheckResponse()  # another replica checked this stretch first

        for check in kept:
            await self.store.add_fact_check(meeting.id, check)
        snippets: list[CodeSnippet] = []
        for found in settled.values():
            snippets += [s for s in found.snippets if s not in snippets]
        cited = {i for c in checks for i in c.snippet_ids}
        return FactCheckResponse(checks=checks, snippets=[s for s in snippets if s.id in cited])

    async def _save(self, state: FactCheckState, since: float | None) -> bool:
        try:
            await self.store.save_fact_check_state_if(state, checked_until=since)
        except Conflict:
            return False
        return True

    async def _check(
        self, meeting: Meeting, labelled: dict[str, TranscriptSegment], sensitivity: Sensitivity
    ) -> dict[str, Settled]:
        """The two model calls, with the lookups in between: claim label -> settled verdict, for
        the claims the model chose to check."""
        llm = self.model()
        tools = ToolOrchestrator(
            llm,
            self.store,
            settings=self.settings,
            memory=self.memory,
            jira_target=self.jira_target,
            github_target=self.github_target,
            max_calls=MAX_LOOKUPS,
            max_evidence=MAX_EVIDENCE,
            timeout=self.timeout,
        )
        toolbox = await tools.toolbox(meeting.team_id, "")
        today = toolbox.today
        code_unavailable = self._code_unavailable(toolbox)
        claims = render_claims(labelled)
        context = [f"Today: {today.isoformat()} ({today:%A})", f'Meeting: "{meeting.title}"']

        plan = await llm.generate_structured(
            render_plan_prompt(context, claims, toolbox, code_unavailable),
            FactCheckPlan,
            system=plan_system(MAX_CLAIMS[sensitivity]),
        )
        planned: dict[str, ClaimPlan] = {}
        for item in plan.claims:
            label = item.claim.strip().strip("[]").lower()
            if label in labelled and label not in planned:
                planned[label] = item
        if not planned:
            return {}

        calls = [
            call for item in planned.values() for call in item.calls if call.tool != OWN_CODE_TOOL
        ]
        _, groups, unavailable = await tools.run(toolbox, calls)
        if code_unavailable is None:
            code_groups, code_errors = await self._code(toolbox, planned.values())
            groups += code_groups
            unavailable += [e for e in code_errors if e not in unavailable]
        evidence, _ = number(groups, MAX_EVIDENCE)
        if not evidence:
            return {
                label: Settled(verdict="unknown", confidence=0.0, severity="low")
                for label in planned
            }

        drafts = await llm.generate_structured(
            render_verdict_prompt(context, claims, planned, evidence, unavailable),
            FactCheckVerdicts,
            system=verdict_system(),
        )
        settled: dict[str, Settled] = {}
        for draft in drafts.checks:
            label = draft.claim.strip().strip("[]").lower()
            if label in planned and label not in settled:
                settled[label] = settle(draft, evidence)
        return settled

    def _code_unavailable(self, toolbox: TeamToolbox) -> str | None:
        if self.code is None:
            return NO_CODE_SEARCH
        if isinstance(toolbox.github, str):
            return toolbox.github
        return None

    async def _code(
        self, toolbox: TeamToolbox, planned: Iterable[ClaimPlan]
    ) -> tuple[list[list[Finding]], list[str]]:
        """Code evidence for the planned code queries, one group per query."""
        assert self.code is not None and isinstance(toolbox.github, GitHubReader)
        queries = list(dict.fromkeys(q.code_query.strip() for q in planned if q.code_query))
        groups: list[list[Finding]] = []
        errors: list[str] = []
        for query in [q for q in queries if q][:MAX_CODE_QUERIES]:
            try:
                with anyio.fail_after(self.timeout):
                    snippets = await self.code(toolbox.github, query)
            except TimeoutError:
                errors.append(f"Code search timed out after {self.timeout:g}s")
                continue
            except Exception as e:
                errors.append(f"Code search failed: {root_cause(e)}")
                continue
            groups.append([code_finding(s) for s in snippets[:MAX_SNIPPETS]])
        return groups, errors


class CodeFinding(Finding):
    snippet: CodeSnippet


def code_finding(snippet: CodeSnippet) -> CodeFinding:
    label = f"{snippet.path}:{snippet.start_line}-{snippet.end_line}"
    text = f"Code {label}: {snippet.caption}\n{snippet.code}"
    source = Source(kind="github_code", label=label, url=snippet.github_url)
    return CodeFinding(text=text, source=source, snippet=snippet)


def settle(draft: ClaimVerdict, evidence: Sequence[Evidence]) -> Settled:
    """Only evidence that exists counts; without any the verdict is unknown."""
    by_id = {e.id: e for e in evidence}
    cited = [i.strip().strip("[]").lower() for i in draft.evidence_ids]
    valid = [by_id[i] for i in dict.fromkeys(cited) if i in by_id]
    if not valid:
        return Settled(verdict="unknown", confidence=0.0, severity=draft.severity)
    sources: list[Source] = []
    snippets: list[CodeSnippet] = []
    for item in valid:
        if item.finding.source not in sources:
            sources.append(item.finding.source)
        if isinstance(item.finding, CodeFinding) and item.finding.snippet not in snippets:
            snippets.append(item.finding.snippet)
    confidence = draft.confidence if math.isfinite(draft.confidence) else 0.0
    return Settled(
        verdict=draft.verdict,
        confidence=min(1.0, max(0.0, confidence)),
        severity=draft.severity,
        finding=clip(draft.finding, MAX_FINDING),
        sources=sources,
        snippets=snippets,
    )


def fact_check(segment: TranscriptSegment, found: Settled) -> FactCheck:
    """Addressed to whoever made the claim."""
    return FactCheck(
        id=str(uuid4()),
        claim=clip(segment.text),
        speaker_name=segment.speaker_name,
        verdict=found.verdict,
        confidence=found.confidence,
        severity=found.severity,
        finding=found.finding,
        snippet_ids=[s.id for s in found.snippets],
        sources=found.sources,
        recipient_id=segment.speaker_id,
        t=segment.t_start,
        created_at=datetime.now(UTC),
    )


# prompts


def plan_system(max_claims: int) -> str:
    agent = get_identity().agent_name
    return f"""You are {agent}, the assistant of a software team. During a live meeting a filter \
picked out claims people made that the team's records might confirm or contradict. Choose the \
claims worth checking and the read-only lookups that would settle each.

Rules:
- Check only concrete, technical claims about the team's work: whether a pull request or issue
  is merged, released, deployed, closed or fixed; Jira status; versions; what the team decided;
  dates and deadlines; configuration values. Skip opinions, plans, jokes and anything the
  records cannot settle.
- Pick at most {max_claims} claims and at most {MAX_LOOKUPS} lookups in all, using only tool names
  from the menu, most useful first. Pick none when nothing is worth checking.
- Merged is not released: for "released", "shipped", "deployed" or "in production", read the
  pull request (github_read with kind pr) and the latest releases (github_releases).
- What the team decided lives in decisions and its past meetings; the live state of issues and
  pull requests lives in Jira and GitHub.
- Set code_query only when code search is available and the claim is about code or settings.
- Search text is short: the key words of the claim, not the whole sentence.
{DATA_RULE}
- You only read. You never create, change or post anything."""


def verdict_system() -> str:
    agent = get_identity().agent_name
    return f"""You are {agent}, the assistant of a software team. You check claims people made \
in a live meeting against numbered evidence from the team's records. Each verdict goes, as a \
short private chat message, only to the person who made the claim; nobody else sees or hears it.

Rules:
- Give one verdict per claim label, using only the numbered evidence.
- contradicted: the evidence clearly shows the claim is wrong. supported: the evidence clearly
  confirms it. unknown: the evidence does not settle it. When in doubt, unknown.
- Merged is not released: a pull request merged after the latest release was published is not
  in any release yet.
- Put the id of every evidence item the verdict relies on in evidence_ids.
- finding is one short, factual sentence on what the evidence shows about the claim, with dates,
  numbers or statuses where they settle it; no evidence ids and no opinion of the speaker.
- confidence is 0 to 1: how sure the evidence makes you.
- severity is high when acting on a wrong claim could cause real harm or a bad decision now:
  announcing or relying on something not released, customer-facing behaviour, security, money,
  data, or a committed deadline. Otherwise low.
{DATA_RULE}
- Judge the claim, never the person."""


def render_claims(labelled: dict[str, TranscriptSegment]) -> list[str]:
    return [
        "Claims ([label] [time] speaker: text):",
        *fenced(
            f"[{label}] [{clock(s.t_start)}] {s.speaker_name}: {clip(s.text)}"
            for label, s in labelled.items()
        ),
    ]


def render_plan_prompt(
    context: list[str], claims: list[str], toolbox: TeamToolbox, code_unavailable: str | None
) -> str:
    lines = [*context, "", *claims, "", f"Tools (at most {MAX_LOOKUPS} lookups in all):"]
    lines += [
        render_tool(spec, toolbox.unavailable(spec.name))
        for spec in toolbox.specs()
        if spec.name != OWN_CODE_TOOL
    ]
    code = "- code search (code_query): search the team's repository's code"
    lines.append(code + (f" [not available: {code_unavailable}]" if code_unavailable else ""))
    return "\n".join(lines)


def render_verdict_prompt(
    context: list[str],
    claims: list[str],
    planned: dict[str, ClaimPlan],
    evidence: list[Evidence],
    unavailable: list[str],
) -> str:
    lines = [*context, "", *claims, "", f"Claims to judge: {', '.join(planned)}", ""]
    lines.append("Evidence ([id] source (date): content):")
    lines += fenced(
        f"[{item.id}] {item.finding.source.label}{dated(item.finding)}: {clip(item.finding.text)}"
        for item in evidence
    )
    if unavailable:
        lines += ["", "Unavailable sources:", *(f"- {reason}" for reason in unavailable)]
    return "\n".join(lines)
