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
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.requests: list[dict] = []

    async def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def gemini(models: FakeModels, model: str = "test-model") -> GeminiLLM:
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return GeminiLLM(model=model, client=client)


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
    models = FakeModels(error=genai_errors.APIError(503, {"error": {"message": "overloaded"}}))

    with pytest.raises(LLMError, match="overloaded"):
        await gemini(models).generate("q")


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
    assert llm.model == "m"
