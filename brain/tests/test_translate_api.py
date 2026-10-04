"""POST /internal/meetings/{id}/translate: the worker's live translation of non-English speech
into English (#106), on the brain's LLM so the worker holds no model key."""

import pytest
from api_support import ALEX, WORKER_TOKEN, create
from fastapi.testclient import TestClient

from brain.api.deps import get_llm_factory
from brain.llm import LLMUnavailable, MockLLM
from brain.translation import Translation


class Translator:
    """A MockLLM whose answer the test sets; it records the prompts it was given."""

    def __init__(self, language: str = "es", english: str = "We keep Postgres for now."):
        self.answer = Translation(language=language, english=english)
        self.llm = MockLLM(structured={Translation: lambda prompt: self.answer})

    @property
    def prompts(self) -> list[str]:
        return [call.prompt for call in self.llm.calls]

    @property
    def systems(self) -> list[str | None]:
        return [call.system for call in self.llm.calls]


@pytest.fixture
def translator(app) -> Translator:
    t = Translator()
    app.dependency_overrides[get_llm_factory] = lambda: lambda: t.llm
    return t


def translate(worker: TestClient, meeting_id: str, text: str, language: str | None = None):
    body = {"text": text} | ({"language": language} if language else {})
    return worker.post(f"/internal/meetings/{meeting_id}/translate", json=body)


def test_speech_in_another_language_comes_back_in_english(worker, client_as, translator):
    meeting = create(client_as(ALEX))

    response = translate(worker, meeting["id"], "Por ahora nos quedamos con Postgres.")

    assert response.status_code == 200, response.text
    assert response.json() == {"language": "es", "text": "We keep Postgres for now."}
    assert "Por ahora nos quedamos con Postgres." in translator.prompts[0]


def test_the_prompt_keeps_names_numbers_and_identifiers_as_said(worker, client_as, translator):
    meeting = create(client_as(ALEX))

    translate(worker, meeting["id"], "Polaris, ¿cómo va DS-104?")

    system = translator.systems[0] or ""
    assert "Polaris" in system
    assert "identifiers" in system


def test_scribes_detected_language_is_trusted_and_given_to_the_model(worker, client_as, translator):
    translator.answer = Translation(language="pt", english="We keep Postgres for now.")
    meeting = create(client_as(ALEX))

    response = translate(worker, meeting["id"], "Por ahora nos quedamos con Postgres.", "es")

    assert response.json()["language"] == "es"
    assert "es" in translator.prompts[0]


def test_english_comes_back_word_for_word(worker, client_as, translator):
    translator.answer = Translation(language="en", english="We will keep Postgres for now.")
    meeting = create(client_as(ALEX))

    response = translate(worker, meeting["id"], "we'll keep postgres for now")

    assert response.json() == {"language": "en", "text": "we'll keep postgres for now"}


@pytest.mark.parametrize(("given", "expected"), [("ES", "es"), (" fr ", "fr"), ("zh-CN", "zh")])
def test_the_models_language_is_normalised_to_an_iso_code(
    worker, client_as, translator, given, expected
):
    translator.answer = Translation(language=given, english="Hello")
    meeting = create(client_as(ALEX))

    assert translate(worker, meeting["id"], "Bonjour").json()["language"] == expected


def test_a_language_the_model_cannot_name_is_a_bad_gateway(worker, client_as, translator):
    translator.answer = Translation(language="Spanish", english="Hello")
    meeting = create(client_as(ALEX))

    assert translate(worker, meeting["id"], "Hola").status_code == 502


def test_no_model_configured_is_unavailable(app, worker, client_as):
    def unavailable():
        raise LLMUnavailable("Gemini is not configured: set GEMINI_API_KEY")

    app.dependency_overrides[get_llm_factory] = lambda: unavailable
    meeting = create(client_as(ALEX))

    response = translate(worker, meeting["id"], "Hola")

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_a_failing_model_is_a_bad_gateway(app, worker, client_as):
    app.dependency_overrides[get_llm_factory] = lambda: lambda: MockLLM()  # nothing scripted
    meeting = create(client_as(ALEX))

    response = translate(worker, meeting["id"], "Hola")

    assert response.status_code == 502


@pytest.mark.parametrize("text", ["   ", "x" * 5001])
def test_blank_or_overlong_text_is_refused(worker, client_as, translator, text):
    meeting = create(client_as(ALEX))

    assert translate(worker, meeting["id"], text).status_code == 422
    assert translator.prompts == []


def test_an_unknown_meeting_is_not_found(worker, translator):
    assert translate(worker, "no-such-meeting", "Hola").status_code == 404


def test_the_worker_token_is_required(app, client_as, translator):
    meeting = create(client_as(ALEX))

    response = TestClient(app).post(
        f"/internal/meetings/{meeting['id']}/translate", json={"text": "Hola"}
    )

    assert response.status_code == 401
    assert WORKER_TOKEN not in response.text
