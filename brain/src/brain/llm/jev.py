"""Jev, TypeSafe's decision model, through OpenRouter's decisions API (JEV_MODEL). It answers
typed questions about a state with probabilities and writes no text: a choice among options, a
noul (the probability that something holds) or a score on an ordered scale. Fast and cheap enough
to ask about every caption; the agenda tracker uses it to label lines and to tell when an item is
over. Prompts and answers are never logged."""

import math
from typing import Any

import httpx

from brain.config import Settings

from .base import LLMError, LLMUnavailable
from .openrouter import RouteError, post

TIMEOUT = httpx.Timeout(5.0, connect=3.0)  # it answers in well under a second


class JevClient:
    """`last_cost` is the latest call's cost in USD, as OpenRouter reports it."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: httpx.Timeout = TIMEOUT,
    ):
        self.model = model
        self.url = url
        self.last_cost: float | None = None
        self._api_key = api_key
        self._transport = transport
        self._timeout = timeout

    async def decide(
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Each question's answer by its key. An error, a timeout or an answer that does not fit
        its question is an LLMError."""
        body = {"model": self.model, "state": state, "questions": questions}
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            try:
                data = await post(client, self.url, self._api_key, body)
            except RouteError as e:
                raise LLMError(f"Jev answered {e.code}: {e.message}") from None
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise LLMError("Jev's reply has no answers")
        for key, question in questions.items():
            check(key, question, answers.get(key))
        usage = data.get("usage")
        self.last_cost = usage.get("cost") if isinstance(usage, dict) else None
        return answers


def check(key: str, question: dict[str, Any], answer: Any) -> None:
    """A choice must be one of the question's options, a noul a probability."""
    if not isinstance(answer, dict):
        raise LLMError(f"Jev left {key} unanswered")
    kind = question.get("type")
    if kind == "choice" and answer.get("choice") not in question.get("criteria", {}):
        raise LLMError(f"Jev's choice for {key} is not one of its options")
    if kind == "noul" and not probability(answer.get("noul")):
        raise LLMError(f"Jev's answer to {key} is not a probability")
    if kind == "score" and not isinstance(answer.get("score"), int | float):
        raise LLMError(f"Jev's score for {key} is not a number")


def probability(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def make_jev(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> JevClient | None:
    """Jev when JEV_MODEL is set, None when it is not. Set without OPENROUTER_API_KEY, which it
    is called with, it is LLMUnavailable."""
    if not settings.jev_model:
        return None
    if not settings.openrouter_api_key:
        raise LLMUnavailable("JEV_MODEL is set, but OPENROUTER_API_KEY, which Jev needs, is not")
    return JevClient(
        model=settings.jev_model,
        api_key=settings.openrouter_api_key,
        url=settings.jev_url,
        transport=transport,
    )
