import math
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from brain.config import Settings
from brain.llm import Embeddings, LLMError, LLMUnavailable, make_embedder
from brain.llm.gemini import GeminiEmbedder
from brain.llm.mock import MockEmbedder

pytestmark = pytest.mark.anyio


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.hypot(*a) * math.hypot(*b))


# MockEmbedder


async def test_mock_embedder_is_deterministic_and_sized():
    embedder = MockEmbedder(dim=32)

    first = await embedder.embed(["refund the affected users", "ship v0.9.4"])
    again = await MockEmbedder(dim=32).embed(["refund the affected users", "ship v0.9.4"])

    assert first == again
    assert first.model == "mock"
    assert [len(v) for v in first.vectors] == [32, 32]


async def test_mock_embedder_puts_texts_sharing_words_closer():
    embedder = MockEmbedder()

    out = await embedder.embed(
        ["Who will refund the users?", "I'll refund the 14 affected users", "Hold the email"],
        task="query",
    )
    query, refund, email = out.vectors

    assert cosine(query, refund) > cosine(query, email)


async def test_mock_embedder_records_texts_and_task():
    embedder = MockEmbedder()

    await embedder.embed(["a"])
    await embedder.embed(["b"], task="query")

    assert embedder.calls == [(["a"], "document"), (["b"], "query")]


# GeminiEmbedder


class FakeEmbedModels:
    """Answers each embed_content request with one vector per text, or raises the error set for
    that model (an exception, or a list consumed one request at a time)."""

    def __init__(self, dim: int = 4, by_model=None, values=None):
        self.dim = dim
        self.by_model = by_model or {}
        self.values = values
        self.requests: list[dict] = []

    async def embed_content(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.by_model.get(kwargs["model"])
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if outcome else None
        if isinstance(outcome, Exception):
            raise outcome
        texts = kwargs["contents"]
        if self.values is not None:
            vectors = self.values(texts)
        else:
            vectors = [[float(len(t)), *([1.0] * (self.dim - 1))] for t in texts]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])

    @property
    def models_called(self) -> list[str]:
        return [r["model"] for r in self.requests]


def embedder(models: FakeEmbedModels, *names: str, dim: int = 4, **kwargs) -> GeminiEmbedder:
    return GeminiEmbedder(
        models=list(names or ["embed-model"]),
        dim=dim,
        client=SimpleNamespace(aio=SimpleNamespace(models=models)),
        **kwargs,
    )


def api_error(code: int, status: str, message: str = "boom") -> genai_errors.APIError:
    return genai_errors.APIError(
        code, {"error": {"code": code, "status": status, "message": message}}
    )


async def test_gemini_embedder_embeds_documents_with_the_document_task_and_dimensions():
    models = FakeEmbedModels(dim=4)

    out = await embedder(models).embed(["one", "three"])

    assert out == Embeddings(model="embed-model", vectors=[[3, 1, 1, 1], [5, 1, 1, 1]])
    (request,) = models.requests
    assert request["model"] == "embed-model"
    assert request["contents"] == ["one", "three"]
    assert request["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert request["config"].output_dimensionality == 4


async def test_gemini_embedder_embeds_queries_with_the_query_task():
    models = FakeEmbedModels()

    await embedder(models).embed(["who refunds?"], task="query")

    assert models.requests[0]["config"].task_type == "RETRIEVAL_QUERY"


async def test_gemini_embedder_sends_batches_and_keeps_order():
    models = FakeEmbedModels()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    out = await embedder(models, batch_size=2).embed(texts)

    assert [r["contents"] for r in models.requests] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert [v[0] for v in out.vectors] == [1, 2, 3, 4, 5]


async def test_gemini_embedder_with_nothing_to_embed_makes_no_request():
    models = FakeEmbedModels()

    out = await embedder(models).embed([])

    assert out == Embeddings(model="embed-model", vectors=[])
    assert models.requests == []


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
async def test_gemini_embedder_falls_back_when_a_model_is_overloaded(code):
    models = FakeEmbedModels(by_model={"primary": api_error(code, "UNAVAILABLE")})

    out = await embedder(models, "primary", "backup").embed(["x"])

    assert models.models_called == ["primary", "backup"]
    assert out.model == "backup"


async def test_gemini_embedder_never_mixes_models_within_one_call():
    # The primary answers the first batch, then is overloaded: every text is re-embedded by the
    # backup, because vectors from different models are not comparable.
    models = FakeEmbedModels(by_model={"primary": [None, api_error(503, "UNAVAILABLE")]})

    out = await embedder(models, "primary", "backup", batch_size=1).embed(["a", "bb"])

    assert models.models_called == ["primary", "primary", "backup", "backup"]
    assert out.model == "backup"
    assert [v[0] for v in out.vectors] == [1, 2]


@pytest.mark.parametrize(("code", "status"), [(400, "INVALID_ARGUMENT"), (404, "NOT_FOUND")])
async def test_gemini_embedder_does_not_fall_back_on_request_errors(code, status):
    models = FakeEmbedModels(by_model={"primary": api_error(code, status, "bad request")})

    with pytest.raises(LLMError, match="bad request"):
        await embedder(models, "primary", "backup").embed(["x"])

    assert models.models_called == ["primary"]


async def test_gemini_embedder_error_names_every_model_tried():
    models = FakeEmbedModels(
        by_model={
            "primary": api_error(503, "UNAVAILABLE"),
            "backup": api_error(429, "RESOURCE_EXHAUSTED"),
        }
    )

    with pytest.raises(LLMError) as info:
        await embedder(models, "primary", "backup").embed(["x"])

    assert "primary: 503 UNAVAILABLE" in str(info.value)
    assert "backup: 429 RESOURCE_EXHAUSTED" in str(info.value)


async def test_gemini_embedder_rejects_vectors_of_the_wrong_size():
    models = FakeEmbedModels(values=lambda texts: [[1.0, 2.0] for _ in texts])

    with pytest.raises(LLMError, match="dimensions"):
        await embedder(models, dim=4).embed(["x"])


async def test_gemini_embedder_rejects_a_missing_vector():
    models = FakeEmbedModels(values=lambda texts: [[1.0] * 4])

    with pytest.raises(LLMError, match="2 texts"):
        await embedder(models).embed(["x", "y"])


def test_gemini_embedder_needs_a_model():
    with pytest.raises(ValueError):
        GeminiEmbedder(models=[], dim=4, client=SimpleNamespace())


@pytest.fixture
def recorded_clients(monkeypatch) -> list[dict]:
    instances: list[dict] = []
    monkeypatch.setattr("brain.llm.gemini.genai.Client", lambda **kwargs: instances.append(kwargs))
    return instances


def test_gemini_embedder_client_fails_fast(recorded_clients):
    GeminiEmbedder(models=["m"], dim=4, api_key="k", attempts=3, max_delay=2.5)

    (kwargs,) = recorded_clients
    assert kwargs["api_key"] == "k"
    retry = kwargs["http_options"].retry_options
    assert (retry.attempts, retry.max_delay) == (3, 2.5)
    assert set(retry.http_status_codes) == {429, 500, 502, 503, 504}


# make_embedder


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


CONFIGURED = {"gemini_api_key": "k", "gemini_embedding_model": "e", "gemini_embedding_dim": 768}


@pytest.mark.parametrize("missing", list(CONFIGURED))
def test_make_embedder_is_unavailable_without_key_model_and_dim(monkeypatch, missing):
    for name in ("GEMINI_API_KEY", "GEMINI_EMBEDDING_MODEL", "GEMINI_EMBEDDING_DIM"):
        monkeypatch.delenv(name, raising=False)
    values = {k: v for k, v in CONFIGURED.items() if k != missing}

    with pytest.raises(LLMUnavailable, match=missing.upper()):
        make_embedder(settings(**values))


def test_make_embedder_builds_gemini_with_fallbacks(recorded_clients):
    out = make_embedder(
        settings(
            **CONFIGURED,
            gemini_embedding_fallback_models="backup, e,,other",
            gemini_attempts=1,
            gemini_max_delay=3,
        )
    )

    assert isinstance(out, GeminiEmbedder)
    assert out.models == ("e", "backup", "other")
    assert out.dim == 768
    retry = recorded_clients[0]["http_options"].retry_options
    assert (retry.attempts, retry.max_delay) == (1, 3)


def test_embedding_fallbacks_default_to_none(monkeypatch):
    monkeypatch.delenv("GEMINI_EMBEDDING_FALLBACK_MODELS", raising=False)

    assert settings().gemini_embedding_fallback_models is None
