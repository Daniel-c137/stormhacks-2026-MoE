"""Link a meeting's new decisions to the team's past decisions they contradict or supersede.

A library step for the post-meeting pipeline. The model only judges candidate pairs it was shown;
a relation is set only for a confident verdict on one of those pairs, never otherwise.
"""

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from brain.llm import LLM
from brain.memory import MemoryHit
from brain.store import Store
from contracts import Decision, DecisionRelation

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 5
"""Past decisions shown to the model per new decision, so the prompt stays small."""

MIN_CONFIDENCE = 0.7

CandidateSource = Callable[[Decision, Sequence[Decision]], list[Decision]]
"""Picks, from eligible past decisions, the ones worth asking about for a new decision."""


class DecisionVerdict(BaseModel):
    decision_id: str = Field(description="Id of the new decision.")
    verdict: Literal["contradicts", "supersedes", "unrelated"] = Field(
        description="'contradicts': the new decision reverses or conflicts with the past one. "
        "'supersedes': it replaces or updates the past one without reversing it. "
        "'unrelated': anything else, including decisions that agree or just share a topic."
    )
    past_decision_id: str | None = Field(
        default=None, description="Id of the listed past decision; null when unrelated."
    )
    reason: str = Field(description="One short sentence, citing what each decision says.")
    confidence: float = Field(ge=0, le=1, description="0 to 1.")


class DecisionVerdicts(BaseModel):
    verdicts: list[DecisionVerdict] = []


class DecisionLinks(BaseModel):
    """Only the decisions that changed: new ones that gained a relation, past ones retired."""

    new: list[Decision] = []
    past: list[Decision] = []


async def link_decisions(
    llm: LLM,
    new: Sequence[Decision],
    past: Sequence[Decision],
    *,
    dates: Mapping[str, date] | None = None,
    candidates: CandidateSource | None = None,
) -> DecisionLinks:
    """`past` must be the same team's decisions; `dates` maps meeting ids to meeting dates."""
    pick = candidates or lexical_candidates
    offered: dict[str, dict[str, Decision]] = {}
    for decision in new:
        eligible = [p for p in past if eligible_pair(decision, p)]
        shown = pick(decision, eligible) if eligible else []
        if shown := [p for p in shown if eligible_pair(decision, p)]:
            offered[decision.id] = {p.id: p for p in shown}
    if not offered:
        return DecisionLinks()

    by_id = {d.id: d for d in new}
    answer = await llm.generate_structured(
        render_prompt([by_id[i] for i in offered], offered, dates or {}),
        DecisionVerdicts,
        system=SYSTEM,
    )

    accepted = sorted(
        (
            (v, offered[v.decision_id][v.past_decision_id])
            for v in answer.verdicts
            if v.verdict != "unrelated"
            and v.confidence >= MIN_CONFIDENCE
            and v.past_decision_id in offered.get(v.decision_id, {})
        ),
        key=lambda pair: pair[0].confidence,
        reverse=True,
    )
    linked: set[str] = set()
    links = DecisionLinks()
    for v, old in accepted:
        if v.decision_id in linked or old.id in linked:
            continue  # one link per decision, the most confident one
        linked |= {v.decision_id, old.id}
        retired = DecisionRelation(type="superseded_by", decision_id=v.decision_id)
        links.past.append(old.model_copy(update={"status": "superseded", "relation": retired}))
        if v.verdict == "contradicts":
            relation = DecisionRelation(type="contradicts", decision_id=old.id)
            links.new.append(by_id[v.decision_id].model_copy(update={"relation": relation}))
    return links


async def apply_links(store: Store, links: DecisionLinks) -> None:
    for decision in [*links.past, *links.new]:
        await store.update_decision(decision)


MEMORY_SEARCH_K = 50
"""Memory chunks read per new decision; most are transcript, summary or task chunks."""


class MemorySearch(Protocol):
    """MeetingMemory's search, all that memory candidates need."""

    async def search(self, team_id: str, query: str, k: int = 8) -> list[MemoryHit]: ...


async def memory_candidates(
    memory: MemorySearch,
    team_id: str,
    new: Sequence[Decision],
    past: Sequence[Decision],
    *,
    k: int = MEMORY_SEARCH_K,
) -> CandidateSource:
    """A candidate source that offers the past decisions meeting memory finds closest to each new
    decision, mapped from decision chunks to `past` by ref_id, then lexical candidates to fill
    the rest. Memory is searched here, once per new decision, because candidate sources are
    synchronous. A failed search leaves that decision with lexical candidates only."""
    known = {p.id for p in past}
    found: dict[str, list[str]] = {}
    for decision in new:
        try:
            hits = await memory.search(team_id, decision.text, k=k)
        except Exception:
            # Memory only ranks candidates; without it the lexical ones still go to the model.
            logger.warning(
                "memory search failed for decision %s; using lexical candidates",
                decision.id,
                exc_info=True,
            )
            hits = []
        refs = [h.chunk.ref_id for h in hits if h.chunk.kind == "decision"]
        found[decision.id] = list(dict.fromkeys(r for r in refs if r in known))

    def pick(decision: Decision, eligible: Sequence[Decision]) -> list[Decision]:
        by_id = {p.id: p for p in eligible}
        ranked = [by_id[i] for i in found.get(decision.id, []) if i in by_id]
        shown = {p.id for p in ranked}
        ranked += [p for p in lexical_candidates(decision, eligible) if p.id not in shown]
        return ranked[:MAX_CANDIDATES]

    return pick


def eligible_pair(new: Decision, past: Decision) -> bool:
    return past.status == "active" and past.id != new.id and past.meeting_id != new.meeting_id


def lexical_candidates(new: Decision, past: Sequence[Decision]) -> list[Decision]:
    """The past decisions sharing the most words with the new one, best first."""
    words = terms(new.text)
    scored = [(len(words & terms(p.text)), i, p) for i, p in enumerate(past)]
    ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))
    return [p for _, _, p in ranked[:MAX_CANDIDATES]]


STOPWORDS = frozenset(
    "a an and are as at be before by for from in into is it its of on or our over the their "
    "them then this to up we will with".split()
)


def terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.casefold()) if w not in STOPWORDS}


SYSTEM = """You compare a software team's new decisions with its earlier decisions.

For each new decision, decide whether it contradicts or supersedes one of the past decisions
listed under it.
- "contradicts": the new decision reverses or conflicts with what the past decision said.
- "supersedes": the new decision replaces or updates the past one without reversing it.
- "unrelated": anything else. Sharing a topic, a technology or a word is not enough.
Use only what the decisions and quotes say. Pick a past decision only from the ones listed
under that new decision, by its id. Give one short reason and a confidence from 0 to 1.
Return one verdict per new decision."""


def render_prompt(
    new: Sequence[Decision],
    offered: Mapping[str, Mapping[str, Decision]],
    dates: Mapping[str, date],
) -> str:
    def line(d: Decision) -> str:
        when = dates.get(d.meeting_id)
        return (
            f"[{d.id}] ({when.isoformat() if when else 'date unknown'}, {d.made_by}) {d.text}\n"
            f'    quote: "{d.quote}"'
        )

    blocks = []
    for d in new:
        blocks += ["New decision:", line(d), "Past decisions to compare with:"]
        blocks += [f"  {line(p)}" for p in offered[d.id].values()]
        blocks.append("")
    return "\n".join(blocks).rstrip()
