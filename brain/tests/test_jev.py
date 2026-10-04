"""Jev, TypeSafe's decision model on OpenRouter: typed questions about a state, answered with
probabilities. The client sends the state and questions as they are and returns the answers; a
failure is an LLMError, never a made-up answer."""

import json

import httpx
import pytest

from brain.config import Settings
from brain.llm import LLMError, LLMUnavailable
from brain.llm.jev import JevClient, make_jev

pytestmark = pytest.mark.anyio

URL = "https://openrouter.ai/api/alpha/decisions"
QUESTIONS = {
    "about": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}},
    "over": {"type": "noul", "instructions": "Over?", "criteria": {"true": "y", "false": "n"}},
}
ANSWERS = {
    "about": {"choice": "a", "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.8},
    "over": {"noul": 0.1},
}


def client(handler, **options) -> JevClient:
    return JevClient(
        model="typesafe/jev-1.13",
        api_key="sk-or-test",
        url=URL,
        transport=httpx.MockTransport(handler),
        **options,
    )


async def test_sends_the_model_state_and_questions_and_returns_the_answers():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"answers": ANSWERS, "usage": {"cost": 0.00004}})

    jev = client(handler)
    answers = await jev.decide({"line": "hello"}, QUESTIONS)

    assert answers == ANSWERS
    [request] = seen
    assert str(request.url) == URL
    assert request.headers["authorization"] == "Bearer sk-or-test"
    assert json.loads(request.content) == {
        "model": "typesafe/jev-1.13",
        "state": {"line": "hello"},
        "questions": QUESTIONS,
    }
    assert jev.last_cost == 0.00004


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_an_error_status_is_an_llm_error_with_its_message(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"code": status, "message": "nope"}})

    with pytest.raises(LLMError, match="nope"):
        await client(handler).decide({}, QUESTIONS)


async def test_a_timeout_is_an_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError, match="timed out"):
        await client(handler).decide({}, QUESTIONS)


@pytest.mark.parametrize(
    "body",
    [
        {"usage": {}},  # no answers
        {"answers": {"about": ANSWERS["about"]}},  # a question left unanswered
        {"answers": {**ANSWERS, "over": {"noul": "high"}}},  # not a probability
        {"answers": {**ANSWERS, "about": {"choice": "c"}}},  # not one of the options
        {"answers": {**ANSWERS, "about": {"choice": ["a"]}}},  # not an option at all
        {"answers": {**ANSWERS, "over": {"noul": True}}},  # not a probability either
    ],
)
async def test_an_answer_that_does_not_fit_the_questions_is_an_llm_error(body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(LLMError):
        await client(handler).decide({}, QUESTIONS)


def test_jev_is_on_only_with_its_model_and_the_openrouter_key():
    def settings(**values) -> Settings:
        return Settings(_env_file=None, **values)

    assert make_jev(settings()) is None
    assert make_jev(settings(openrouter_api_key="sk-or-test")) is None
    with pytest.raises(LLMUnavailable, match="OPENROUTER_API_KEY"):
        make_jev(settings(jev_model="typesafe/jev-1.13"))
    jev = make_jev(settings(jev_model="typesafe/jev-1.13", openrouter_api_key="sk-or-test"))
    assert isinstance(jev, JevClient) and jev.model == "typesafe/jev-1.13"
