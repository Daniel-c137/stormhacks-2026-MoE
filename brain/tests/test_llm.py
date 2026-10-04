from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from brain.config import Settings
from brain.llm import LLMError, LLMUnavailable, make_llm
from brain.llm.gemini import GeminiLLM
from brain.llm.mock import MockLLM

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
    monkeypatch.setattr("brain.llm.gemini.warned_missing", set(), raising=False)


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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    with pytest.raises(LLMUnavailable, match="GEMINI_"):
        make_llm(settings(**values))


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


async def test_mock_reports_itself_as_the_model():
    llm = MockLLM(text="hello")

    await llm.generate("hi")

    assert llm.last_model == "mock"
