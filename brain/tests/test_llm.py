import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from brain.config import Settings
from brain.llm import (
    FallbackLLM,
    LLMError,
    LLMOutOfCapacity,
    LLMUnavailable,
    make_llm,
    make_translation_llm,
)
from brain.llm.gemini import GeminiLLM
from brain.llm.mock import MockLLM
from brain.llm.openrouter import OpenRouterLLM

pytestmark = pytest.mark.anyio


class Verdict(BaseModel):
    answer: str
    confidence: float


# MockLLM


async def test_mock_returns_canned_structured_response_and_records_the_call():
    llm = MockLLM(structured={Verdict: Verdict(answer="yes", confidence=0.9)})

    out = await llm.generate_structured("Is it fixed?", Verdict, system="Be brief.")

    assert out == Verdict(answer="yes", confidence=0.9)
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert (call.prompt, call.system, call.schema) == ("Is it fixed?", "Be brief.", Verdict)


async def test_mock_computes_a_response_from_the_prompt():
    llm = MockLLM(structured={Verdict: lambda prompt: Verdict(answer=prompt.upper(), confidence=1)})

    out = await llm.generate_structured("merged", Verdict)

    assert out.answer == "MERGED"


async def test_mock_hands_out_copies_so_callers_cannot_change_the_script():
    llm = MockLLM(structured={Verdict: Verdict(answer="yes", confidence=0.9)})

    first = await llm.generate_structured("q", Verdict)
    first.answer = "changed"

    assert (await llm.generate_structured("q", Verdict)).answer == "yes"


async def test_mock_without_a_response_for_the_schema_fails_loudly():
    llm = MockLLM()

    with pytest.raises(LLMError, match="Verdict"):
        await llm.generate_structured("q", Verdict)


async def test_mock_generates_text():
    llm = MockLLM(text="hello")

    assert await llm.generate("hi", system="s") == "hello"
    assert llm.calls[0].schema is None


# GeminiLLM


class FakeModels:
    """Each model answers with a response or raises an exception; `by_model` overrides the
    outcome shared by every model."""

    def __init__(self, response=None, error: Exception | None = None, by_model=None):
        self.default = error or response
        self.by_model = by_model or {}
        self.requests: list[dict] = []

    async def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.by_model.get(kwargs["model"], self.default)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def models_called(self) -> list[str]:
        return [r["model"] for r in self.requests]


def fake_client(models: FakeModels):
    return SimpleNamespace(aio=SimpleNamespace(models=models))


def gemini(models: FakeModels, *names: str) -> GeminiLLM:
    return GeminiLLM(models=list(names or ["test-model"]), client=fake_client(models))


def api_error(code: int, status: str, message: str = "boom") -> genai_errors.APIError:
    body = {"error": {"code": code, "status": status, "message": message}}
    return genai_errors.APIError(code, body)


@pytest.fixture(autouse=True)
def forget_missing_models(monkeypatch):
    """Each test starts as a fresh process that has not yet warned about any missing model."""
    monkeypatch.setattr("brain.llm.gemini.warned_missing", set())


@pytest.fixture
def recorded_clients(monkeypatch) -> list[dict]:
    """Replaces genai.Client and records the arguments GeminiLLM builds it with."""
    instances: list[dict] = []
    monkeypatch.setattr("brain.llm.gemini.genai.Client", lambda **kwargs: instances.append(kwargs))
    return instances


def response(text: str | None, parsed=None):
    return SimpleNamespace(text=text, parsed=parsed)


async def test_gemini_requests_schema_constrained_json_and_parses_it():
    models = FakeModels(response('{"answer": "no", "confidence": 0.4}'))

    out = await gemini(models).generate_structured("Is it live?", Verdict, system="Rules")

    assert out == Verdict(answer="no", confidence=0.4)
    request = models.requests[0]
    assert request["model"] == "test-model"
    assert request["contents"] == "Is it live?"
    config = request["config"]
    assert config.system_instruction == "Rules"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is Verdict


async def test_gemini_uses_the_sdk_parsed_object_when_present():
    parsed = Verdict(answer="parsed", confidence=1)
    models = FakeModels(response("ignored", parsed=parsed))

    assert await gemini(models).generate_structured("q", Verdict) == parsed


async def test_gemini_generates_text():
    models = FakeModels(response("plain answer"))

    assert await gemini(models).generate("q", system="s") == "plain answer"
    assert models.requests[0]["config"].system_instruction == "s"


async def test_gemini_response_that_does_not_match_the_schema_is_an_llm_error():
    models = FakeModels(response('{"answer": "no"}'))

    with pytest.raises(LLMError, match="Verdict"):
        await gemini(models).generate_structured("q", Verdict)


async def test_gemini_empty_response_is_an_llm_error():
    models = FakeModels(response(None))

    with pytest.raises(LLMError, match="no content"):
        await gemini(models).generate_structured("q", Verdict)


async def test_gemini_api_failure_is_an_llm_error():
    models = FakeModels(error=api_error(503, "UNAVAILABLE", "overloaded"))

    with pytest.raises(LLMError, match="overloaded"):
        await gemini(models).generate("q")


async def test_gemini_reports_the_model_that_answered():
    llm = gemini(FakeModels(response("hi")), "primary", "backup")

    assert llm.last_model is None
    await llm.generate("q")

    assert llm.last_model == "primary"


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (429, "RESOURCE_EXHAUSTED"),
        (500, "INTERNAL"),
        (502, "BAD_GATEWAY"),
        (503, "UNAVAILABLE"),
        (504, "DEADLINE_EXCEEDED"),
    ],
)
async def test_gemini_falls_back_in_order_when_a_model_is_overloaded(code, status):
    models = FakeModels(
        by_model={
            "primary": api_error(code, status),
            "second": api_error(503, "UNAVAILABLE"),
            "third": response('{"answer": "ok", "confidence": 1}'),
        }
    )
    llm = gemini(models, "primary", "second", "third")

    out = await llm.generate_structured("q", Verdict)

    assert out.answer == "ok"
    assert models.models_called == ["primary", "second", "third"]
    assert llm.last_model == "third"


async def test_gemini_stops_at_the_first_model_that_answers():
    models = FakeModels(response("hi"))

    await gemini(models, "primary", "backup").generate("q")

    assert models.models_called == ["primary"]


@pytest.mark.parametrize(
    ("code", "status"),
    [(400, "INVALID_ARGUMENT"), (401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED")],
)
async def test_gemini_does_not_fall_back_on_request_errors(code, status):
    models = FakeModels(by_model={"primary": api_error(code, status, "bad request")})
    llm = gemini(models, "primary", "backup")

    with pytest.raises(LLMError, match="bad request"):
        await llm.generate("q")

    assert models.models_called == ["primary"]
    assert llm.last_model is None


async def test_gemini_skips_a_model_that_does_not_exist_for_this_key():
    models = FakeModels(
        by_model={
            "retired": api_error(404, "NOT_FOUND", "no longer available"),
            "working": response("from the working model"),
        }
    )
    llm = gemini(models, "retired", "working")

    assert await llm.generate("q") == "from the working model"

    assert models.models_called == ["retired", "working"]
    assert llm.last_model == "working"


async def test_gemini_error_names_every_missing_model():
    models = FakeModels(error=api_error(404, "NOT_FOUND", "no longer available"))

    with pytest.raises(LLMError, match="unavailable on every model tried") as info:
        await gemini(models, "retired", "unknown").generate("q")

    message = str(info.value)
    assert "retired: 404 NOT_FOUND" in message
    assert "unknown: 404 NOT_FOUND" in message
    assert models.models_called == ["retired", "unknown"]


async def test_gemini_stops_at_a_request_error_after_a_missing_model():
    models = FakeModels(
        by_model={
            "retired": api_error(404, "NOT_FOUND"),
            "second": api_error(400, "INVALID_ARGUMENT", "bad schema"),
        }
    )

    with pytest.raises(LLMError, match="bad schema"):
        await gemini(models, "retired", "second", "third").generate("q")

    assert models.models_called == ["retired", "second"]


async def test_gemini_warns_once_per_missing_model_without_the_request(caplog):
    models = FakeModels(
        by_model={"stale-model": api_error(404, "NOT_FOUND"), "working": response("ok")}
    )
    llm = gemini(models, "stale-model", "working")

    with caplog.at_level("WARNING", logger="brain.llm.gemini"):
        await llm.generate("secret meeting question", system="secret system prompt")
        await llm.generate("secret meeting question")
        await gemini(models, "stale-model", "working").generate("another question")

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "stale-model" in warnings[0]
    assert "secret" not in warnings[0]
    assert "question" not in warnings[0]


async def test_gemini_does_not_fall_back_on_unusable_content():
    models = FakeModels(by_model={"primary": response('{"answer": "no"}')})

    with pytest.raises(LLMError, match="Verdict"):
        await gemini(models, "primary", "backup").generate_structured("q", Verdict)

    assert models.models_called == ["primary"]


async def test_gemini_error_names_every_model_tried_with_its_status():
    models = FakeModels(
        by_model={
            "primary": api_error(503, "UNAVAILABLE"),
            "backup": api_error(429, "RESOURCE_EXHAUSTED"),
        }
    )

    with pytest.raises(LLMError) as info:
        await gemini(models, "primary", "backup").generate("q")

    message = str(info.value)
    assert "primary: 503 UNAVAILABLE" in message
    assert "backup: 429 RESOURCE_EXHAUSTED" in message
    assert message.index("primary") < message.index("backup")


def test_gemini_needs_at_least_one_model():
    with pytest.raises(ValueError):
        GeminiLLM(models=[], client=fake_client(FakeModels()))


def test_gemini_client_fails_fast_with_the_configured_attempts(recorded_clients):
    GeminiLLM(models=["m"], api_key="k", attempts=3, max_delay=2.5)

    (kwargs,) = recorded_clients
    assert kwargs["api_key"] == "k"
    retry = kwargs["http_options"].retry_options
    assert retry.attempts == 3
    assert retry.max_delay == 2.5
    assert set(retry.http_status_codes) == {429, 500, 502, 503, 504}


def test_gemini_client_tries_each_model_twice_by_default(recorded_clients):
    GeminiLLM(models=["m"], api_key="k")

    retry = recorded_clients[0]["http_options"].retry_options
    assert retry.attempts == 2
    assert retry.max_delay <= 5


# make_llm


def settings(**values) -> Settings:
    """Only the given values: no .env, and no OpenRouter unless asked for."""
    values = {"openrouter_api_key": None, "openrouter_models": None} | values
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize(
    "values",
    [
        {"gemini_model": "m"},
        {"gemini_api_key": "k"},
        {},
    ],
)
def test_make_llm_is_unavailable_without_key_and_model(monkeypatch, values):
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "OPENROUTER_API_KEY", "OPENROUTER_MODELS"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(LLMUnavailable, match="GEMINI_") as info:
        make_llm(settings(**values))

    message = str(info.value)
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "OPENROUTER_API_KEY", "OPENROUTER_MODELS"):
        assert name in message


def test_make_llm_builds_gemini_from_config():
    llm = make_llm(settings(gemini_api_key="k", gemini_model="m"))

    assert isinstance(llm, GeminiLLM)
    assert llm.models == ("m",)


def test_make_llm_appends_fallback_models_after_the_primary():
    llm = make_llm(
        settings(
            gemini_api_key="k",
            gemini_model="primary",
            gemini_fallback_models=" second, ,primary,third,second ",
        )
    )

    assert isinstance(llm, GeminiLLM)
    assert llm.models == ("primary", "second", "third")


def test_make_llm_passes_the_retry_settings_to_the_client(recorded_clients):
    make_llm(settings(gemini_api_key="k", gemini_model="m", gemini_attempts=1, gemini_max_delay=3))

    retry = recorded_clients[0]["http_options"].retry_options
    assert (retry.attempts, retry.max_delay) == (1, 3)


def test_retry_settings_default_to_two_attempts_and_no_fallbacks(monkeypatch):
    for name in ("GEMINI_ATTEMPTS", "GEMINI_MAX_DELAY", "GEMINI_FALLBACK_MODELS"):
        monkeypatch.delenv(name, raising=False)

    s = settings()

    assert s.gemini_attempts == 2
    assert s.gemini_max_delay <= 5
    assert s.gemini_fallback_models is None


# translation (#106 review): cheap, single attempt, never a paid fallback on a quota error


def test_translation_runs_on_its_own_model_with_one_attempt_and_no_fallback(recorded_clients):
    llm = make_translation_llm(
        settings(
            gemini_api_key="k",
            gemini_model="main",
            gemini_fallback_models="other",
            translation_model="lite",
            openrouter_api_key="or-key",
            openrouter_models="some/model",
        )
    )

    assert isinstance(llm, GeminiLLM)  # not a FallbackLLM: no OpenRouter bill for a caption
    assert llm.models == ("lite",)
    assert recorded_clients[-1]["http_options"].retry_options.attempts == 1


def test_translation_uses_the_main_gemini_model_unless_given_its_own():
    llm = make_translation_llm(settings(gemini_api_key="k", gemini_model="main"))

    assert isinstance(llm, GeminiLLM)
    assert llm.models == ("main",)


def test_translation_uses_openrouter_only_when_there_is_no_gemini():
    llm = make_translation_llm(settings(openrouter_api_key="or-key", openrouter_models="a/b"))

    assert isinstance(llm, OpenRouterLLM)


def test_translation_is_unavailable_without_any_model():
    with pytest.raises(LLMUnavailable):
        make_translation_llm(settings())


async def test_mock_reports_itself_as_the_model():
    llm = MockLLM(text="hello")

    await llm.generate("hi")

    assert llm.last_model == "mock"


# out of capacity: what lets the provider chain move on


async def test_gemini_with_every_model_overloaded_or_missing_is_out_of_capacity():
    models = FakeModels(
        by_model={"primary": api_error(429, "RESOURCE_EXHAUSTED"), "retired": api_error(404, "X")}
    )

    with pytest.raises(LLMOutOfCapacity):
        await gemini(models, "primary", "retired").generate("q")


@pytest.mark.parametrize("code", [400, 401, 403])
async def test_gemini_request_error_is_not_out_of_capacity(code):
    models = FakeModels(error=api_error(code, "INVALID_ARGUMENT"))

    with pytest.raises(LLMError) as info:
        await gemini(models, "primary", "backup").generate("q")

    assert not isinstance(info.value, LLMOutOfCapacity)


async def test_gemini_unusable_content_is_not_out_of_capacity():
    models = FakeModels(response('{"answer": "no"}'))

    with pytest.raises(LLMError) as info:
        await gemini(models).generate_structured("q", Verdict)

    assert not isinstance(info.value, LLMOutOfCapacity)


# the provider chain: Gemini, then OpenRouter


class FakeRouter:
    """OpenRouter answering every model with `outcome`: a JSON body or an error status."""

    def __init__(self, outcome: dict | int):
        self.outcome = outcome
        self.models_called: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.models_called.append(json.loads(request.content)["model"])
        if isinstance(self.outcome, int):
            return httpx.Response(self.outcome, json={"error": {"code": self.outcome}})
        return httpx.Response(200, json=self.outcome)


def routed(content: str) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def openrouter(router: FakeRouter, *names: str) -> OpenRouterLLM:
    return OpenRouterLLM(
        list(names or ["vendor/model"]),
        api_key="k",
        base_url="https://router.test/api/v1",
        transport=httpx.MockTransport(router.handle),
    )


ROUTED_VERDICT = routed('{"answer": "from openrouter", "confidence": 1}')
GEMINI_VERDICT = response('{"answer": "from gemini", "confidence": 1}')


async def test_chain_uses_gemini_while_it_answers():
    models, router = FakeModels(GEMINI_VERDICT), FakeRouter(ROUTED_VERDICT)
    llm = FallbackLLM([gemini(models, "g"), openrouter(router, "vendor/model")])

    out = await llm.generate_structured("q", Verdict)

    assert out.answer == "from gemini"
    assert router.models_called == []
    assert llm.last_model == "g"


@pytest.mark.parametrize(
    ("code", "status"), [(429, "RESOURCE_EXHAUSTED"), (503, "UNAVAILABLE"), (404, "NOT_FOUND")]
)
async def test_chain_falls_through_to_openrouter_when_gemini_is_out_of_capacity(code, status):
    models = FakeModels(error=api_error(code, status))
    router = FakeRouter(ROUTED_VERDICT)
    llm = FallbackLLM([gemini(models, "g1", "g2"), openrouter(router, "vendor/model")])

    out = await llm.generate_structured("q", Verdict)

    assert out.answer == "from openrouter"
    assert models.models_called == ["g1", "g2"]
    assert router.models_called == ["vendor/model"]
    assert llm.last_model == "openrouter:vendor/model"


async def test_chain_falls_through_for_text_too():
    models = FakeModels(error=api_error(429, "RESOURCE_EXHAUSTED"))
    llm = FallbackLLM([gemini(models), openrouter(FakeRouter(routed("plain")))])

    assert await llm.generate("q") == "plain"


@pytest.mark.parametrize(("code", "status"), [(400, "INVALID_ARGUMENT"), (401, "UNAUTHENTICATED")])
async def test_chain_does_not_fall_through_on_a_gemini_request_error(code, status):
    models = FakeModels(error=api_error(code, status, "bad request"))
    router = FakeRouter(ROUTED_VERDICT)
    llm = FallbackLLM([gemini(models), openrouter(router)])

    with pytest.raises(LLMError, match="bad request"):
        await llm.generate_structured("q", Verdict)

    assert router.models_called == []
    assert llm.last_model is None


async def test_chain_does_not_fall_through_on_unusable_gemini_content():
    router = FakeRouter(ROUTED_VERDICT)
    llm = FallbackLLM([gemini(FakeModels(response('{"answer": 1}'))), openrouter(router)])

    with pytest.raises(LLMError, match="Verdict"):
        await llm.generate_structured("q", Verdict)

    assert router.models_called == []


async def test_chain_with_every_provider_out_of_capacity_names_both():
    models = FakeModels(error=api_error(429, "RESOURCE_EXHAUSTED"))
    llm = FallbackLLM([gemini(models, "g"), openrouter(FakeRouter(503), "vendor/model")])

    with pytest.raises(LLMOutOfCapacity) as info:
        await llm.generate("q")

    message = str(info.value)
    assert "g: 429" in message
    assert "vendor/model: 503" in message
    assert llm.last_model is None


async def test_chain_resets_the_answering_model_on_each_call():
    gemini_models = FakeModels(GEMINI_VERDICT)
    llm = FallbackLLM([gemini(gemini_models, "g"), openrouter(FakeRouter(ROUTED_VERDICT))])
    await llm.generate_structured("q", Verdict)
    gemini_models.default = api_error(429, "RESOURCE_EXHAUSTED")

    await llm.generate_structured("q", Verdict)

    assert llm.last_model == "openrouter:vendor/model"


async def test_chain_logs_the_fall_through_without_the_prompt(caplog):
    models = FakeModels(error=api_error(429, "RESOURCE_EXHAUSTED"))
    llm = FallbackLLM([gemini(models), openrouter(FakeRouter(ROUTED_VERDICT))])

    with caplog.at_level("DEBUG"):
        await llm.generate_structured("secret meeting question", Verdict, system="secret rules")

    logged = [r.getMessage() for r in caplog.records]
    assert any("429" in m for m in logged)
    assert not any("secret" in m for m in logged)


def test_chain_needs_a_provider():
    with pytest.raises(ValueError):
        FallbackLLM([])


# make_llm with OpenRouter


OPENROUTER = {
    "openrouter_api_key": "or-key",
    "openrouter_models": " vendor/first, ,vendor/second,vendor/first ",
}


def test_make_llm_chains_gemini_then_openrouter():
    llm = make_llm(settings(gemini_api_key="k", gemini_model="g", **OPENROUTER))

    assert isinstance(llm, FallbackLLM)
    first, second = llm.providers
    assert isinstance(first, GeminiLLM) and first.models == ("g",)
    assert isinstance(second, OpenRouterLLM)
    assert second.models == ("vendor/first", "vendor/second")


def test_make_llm_with_only_openrouter_uses_it():
    llm = make_llm(settings(**OPENROUTER))

    assert isinstance(llm, OpenRouterLLM)
    assert llm.models == ("vendor/first", "vendor/second")
    assert llm.endpoint == "https://openrouter.ai/api/v1/chat/completions"


def test_make_llm_with_a_gemini_key_only_for_embeddings_uses_openrouter():
    llm = make_llm(settings(gemini_api_key="k", **OPENROUTER))

    assert isinstance(llm, OpenRouterLLM)


def test_make_llm_uses_the_configured_openrouter_url():
    llm = make_llm(settings(openrouter_url="https://proxy.test/v1", **OPENROUTER))

    assert isinstance(llm, OpenRouterLLM)
    assert llm.endpoint == "https://proxy.test/v1/chat/completions"


@pytest.mark.parametrize(
    "values",
    [{"openrouter_api_key": "or-key"}, {"openrouter_models": "vendor/model"}],
)
def test_make_llm_with_only_gemini_complete_uses_gemini(values):
    llm = make_llm(settings(gemini_api_key="k", gemini_model="g", **values))

    assert isinstance(llm, GeminiLLM)


@pytest.mark.parametrize(
    "values",
    [
        {"openrouter_api_key": "or-key"},
        {"openrouter_models": "vendor/model"},
        {"openrouter_api_key": "or-key", "openrouter_models": " , "},
    ],
)
def test_make_llm_with_half_an_openrouter_config_is_unavailable(values):
    with pytest.raises(LLMUnavailable, match="OPENROUTER_MODELS"):
        make_llm(settings(**values))


def test_openrouter_settings_default_to_the_public_api_and_no_models(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODELS", "OPENROUTER_URL"):
        monkeypatch.delenv(name, raising=False)

    s = Settings(_env_file=None)

    assert s.openrouter_url == "https://openrouter.ai/api/v1"
    assert s.openrouter_api_key is None and s.openrouter_models is None
