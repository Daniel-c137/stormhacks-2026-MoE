"""Scripting the agent's two model calls: a plan of tool calls, then an answer that cites the
evidence the prompt numbered."""

import re
from collections.abc import Callable

from brain.agent.ask import AskPlan, DraftAnswer, PlannedCall
from brain.llm import MockLLM

EVIDENCE_LINE = re.compile(r"^\[(e\d+)\] (.*)$", re.MULTILINE)


def evidence(prompt: str) -> dict[str, str]:
    """The numbered evidence the answer prompt showed: id -> its line."""
    return dict(EVIDENCE_LINE.findall(prompt))


def ids_for(prompt: str, *needles: str) -> list[str]:
    """Ids of the evidence lines that contain each needle, in needle order."""
    found = evidence(prompt)
    ids = []
    for needle in needles:
        matches = [i for i, line in found.items() if needle in line]
        assert matches, f"no evidence line contains {needle!r}: {found}"
        ids.append(matches[0])
    return ids


def citing(
    *needles: str, text: str = "Answer.", extra_ids=(), inference: str | None = None
) -> Callable[[str], DraftAnswer]:
    """An answer citing the evidence lines that contain the needles, plus `extra_ids` as is."""

    def answer(prompt: str) -> DraftAnswer:
        ids = [*ids_for(prompt, *needles), *extra_ids]
        return DraftAnswer(text=text, evidence_ids=ids, inference=inference)

    return answer


def scripted(*calls: PlannedCall, answer=None) -> MockLLM:
    return MockLLM(
        structured={
            AskPlan: AskPlan(calls=list(calls)),
            DraftAnswer: answer or DraftAnswer(text="Answer.", evidence_ids=[]),
        }
    )
