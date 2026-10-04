import json
import logging
from collections.abc import Callable

import httpx
import pytest
from pydantic import BaseModel

from brain.llm import LLMError, LLMOutOfCapacity
from brain.llm.openrouter import MAX_TOKENS, OpenRouterLLM

pytestmark = pytest.mark.anyio

BASE_URL = "https://router.test/api/v1"
SECRET_PROMPT = "secret meeting transcript about the acquisition"
SECRET_SYSTEM = "secret system prompt"


class Verdict(BaseModel):
    answer: str
    confidence: float


def completion(content: str | None, *, finish_reason: str = "stop") -> dict:
    return {
        "id": "gen-1",
        "model": "served/model-20261001",
        "provider": "SomeProvider",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 129, "cost": 3e-5},
    }


def failure(code: int, message: str = "boom") -> httpx.Response:
    return httpx.Response(code, json={"error": {"code": code, "message": message}})


Outcome = dict | httpx.Response | Exception


class FakeRouter:
    """OpenRouter's chat completions: each model answers with a completion body, an error
    response, or raises a transport error; `by_model` overrides the outcome shared by every
    model."""

    def __init__(self, outcome: Outcome | None = None, by_model: dict[str, Outcome] | None = None):
        self.default = outcome
        self.by_model = by_model or {}
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        model = json.loads(request.content)["model"]
        outcome = self.by_model.get(model, self.default)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, httpx.Response):
            return outcome
        return httpx.Response(200, json=outcome)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]

    @property
    def models_called(self) -> list[str]:
        return [b["model"] for b in self.bodies]


def router_llm(router: FakeRouter, *models: str) -> OpenRouterLLM:
    return OpenRouterLLM(
        list(models or ["vendor/model"]),
        api_key="test-key",
        base_url=BASE_URL,
        transport=httpx.MockTransport(router.handle),
    )


VALID = completion('{"answer": "yes", "confidence": 0.9}')


# the request


async def test_structured_request_asks_for_the_schema_from_providers_that_keep_no_data():
    router = FakeRouter(VALID)

    await router_llm(router, "vendor/model").generate_structured(
        "Is it fixed?", Verdict, system="Be brief."
    )

    (request,) = router.requests
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = router.bodies[0]
    assert body["model"] == "vendor/model"
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Is it fixed?"},
    ]
    assert body["max_tokens"] == MAX_TOKENS
    assert body["provider"]["require_parameters"] is True
    assert body["provider"]["data_collection"] == "deny"
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Verdict"
    assert response_format["json_schema"]["schema"] == Verdict.model_json_schema()
    assert isinstance(response_format["json_schema"]["strict"], bool)


def test_max_tokens_is_bounded():
    assert 1000 <= MAX_TOKENS <= 16_000


async def test_text_request_has_no_response_format_but_the_same_provider_rules():
    router = FakeRouter(completion("plain answer"))

    assert await router_llm(router).generate("q") == "plain answer"

    body = router.bodies[0]
    assert "response_format" not in body
    assert body["messages"] == [{"role": "user", "content": "q"}]
    assert body["provider"]["require_parameters"] is True
    assert body["provider"]["data_collection"] == "deny"
    assert body["max_tokens"] == MAX_TOKENS


def test_base_url_may_end_with_a_slash():
    router = FakeRouter(VALID)
    llm = OpenRouterLLM(
        ["m"],
        api_key="k",
        base_url=BASE_URL + "/",
        transport=httpx.MockTransport(router.handle),
    )

    assert llm.endpoint == f"{BASE_URL}/chat/completions"


def test_needs_at_least_one_model():
    with pytest.raises(ValueError):
        OpenRouterLLM([], api_key="k", base_url=BASE_URL)


# the response


async def test_structured_response_parses_into_the_schema():
    out = await router_llm(FakeRouter(VALID)).generate_structured("q", Verdict)

    assert out == Verdict(answer="yes", confidence=0.9)


async def test_structured_response_in_a_code_fence_still_parses():
    fenced = completion('```json\n{"answer": "yes", "confidence": 1}\n```')

    out = await router_llm(FakeRouter(fenced)).generate_structured("q", Verdict)

    assert out == Verdict(answer="yes", confidence=1)


@pytest.mark.parametrize(
    "content",
    ['{"answer": "secret answer"}', "not json at all, secret", '["secret"]'],
)
async def test_response_that_does_not_match_the_schema_is_an_llm_error(content):
    router = FakeRouter(completion(content))

    with pytest.raises(LLMError, match="Verdict") as info:
        await router_llm(router, "first", "second").generate_structured("q", Verdict)

    assert not isinstance(info.value, LLMOutOfCapacity)
    assert "secret" not in str(info.value)
    assert router.models_called == ["first"]


async def test_empty_response_is_an_llm_error():
    with pytest.raises(LLMError, match="no content"):
        await router_llm(FakeRouter(completion(None))).generate_structured("q", Verdict)


async def test_response_cut_off_at_max_tokens_is_an_llm_error():
    cut = completion('{"answer": "ye', finish_reason="length")

    with pytest.raises(LLMError, match="max_tokens"):
        await router_llm(FakeRouter(cut)).generate_structured("q", Verdict)


async def test_reports_the_model_that_answered_and_the_usage():
    router = FakeRouter(by_model={"busy/model": failure(429), "spare/model": VALID})
    llm = router_llm(router, "busy/model", "spare/model")

    assert llm.last_model is None
    await llm.generate_structured("q", Verdict)

    assert llm.last_model == "openrouter:spare/model"
    assert llm.last_usage is not None
    assert llm.last_usage["cost"] == 3e-5
    assert llm.last_usage["completion_tokens"] == 9


# the model chain


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 404])
async def test_moves_to_the_next_model_on_overload_or_a_missing_model(code):
    router = FakeRouter(by_model={"first": failure(code), "second": failure(503), "third": VALID})
    llm = router_llm(router, "first", "second", "third")

    out = await llm.generate_structured("q", Verdict)

    assert out.answer == "yes"
    assert router.models_called == ["first", "second", "third"]
    assert llm.last_model == "openrouter:third"


async def test_moves_to_the_next_model_on_a_timeout():
    router = FakeRouter(by_model={"slow": httpx.ReadTimeout("timed out"), "fast": VALID})

    await router_llm(router, "slow", "fast").generate_structured("q", Verdict)

    assert router.models_called == ["slow", "fast"]


async def test_moves_on_when_a_200_carries_an_overload_error():
    """OpenRouter may answer 200 with the provider's error in the body."""
    errored = {"error": {"code": 502, "message": "Provider returned error"}}
    router = FakeRouter(by_model={"first": errored, "second": VALID})

    await router_llm(router, "first", "second").generate_structured("q", Verdict)

    assert router.models_called == ["first", "second"]


async def test_stops_at_the_first_model_that_answers():
    router = FakeRouter(VALID)

    await router_llm(router, "first", "second").generate("q")

    assert router.models_called == ["first"]


@pytest.mark.parametrize("code", [400, 401, 402, 403])
async def test_does_not_move_on_after_a_request_error(code):
    router = FakeRouter(failure(code, "bad request"))
    llm = router_llm(router, "first", "second")

    with pytest.raises(LLMError, match="bad request") as info:
        await llm.generate_structured("q", Verdict)

    assert not isinstance(info.value, LLMOutOfCapacity)
    assert router.models_called == ["first"]
    assert llm.last_model is None


async def test_every_model_out_of_capacity_is_out_of_capacity_naming_each_model():
    router = FakeRouter(by_model={"first": failure(429, "rate limited"), "second": failure(404)})

    with pytest.raises(LLMOutOfCapacity) as info:
        await router_llm(router, "first", "second").generate("q")

    message = str(info.value)
    assert "OpenRouter" in message
    assert "first: 429" in message and "rate limited" in message
    assert "second: 404" in message
    assert message.index("first") < message.index("second")


async def test_the_api_key_never_appears_in_errors():
    router = FakeRouter(failure(401, "No auth credentials found"))

    with pytest.raises(LLMError) as info:
        await router_llm(router).generate("q")

    assert "test-key" not in str(info.value)


# privacy


async def test_no_prompt_or_response_text_in_errors_or_logs(caplog):
    outcomes: list[Callable[[], FakeRouter]] = [
        lambda: FakeRouter(failure(429)),
        lambda: FakeRouter(by_model={"m": failure(404)}),
        lambda: FakeRouter(failure(400, "invalid request")),
        lambda: FakeRouter(completion('{"answer": "secret reply"}')),
        lambda: FakeRouter(VALID),
    ]
    messages: list[str] = []
    with caplog.at_level(logging.DEBUG):
        for make in outcomes:
            try:
                await router_llm(make(), "m").generate_structured(
                    SECRET_PROMPT, Verdict, system=SECRET_SYSTEM
                )
            except LLMError as e:
                messages.append(str(e))

    assert len(messages) == 4
    for text in [*messages, *(r.getMessage() for r in caplog.records)]:
        assert "secret" not in text
        assert "acquisition" not in text
