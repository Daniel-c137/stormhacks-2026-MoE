import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from brain.config import Settings
from brain.llm import (
    Embeddings,
    FallbackEmbedder,
    LLMError,
    LLMOutOfCapacity,
    LLMUnavailable,
    make_embedder,
)
from brain.llm.gemini import GeminiEmbedder
from brain.llm.openrouter import OpenRouterEmbedder
from brain.memory import Chunk, InMemoryMemoryStore

pytestmark = pytest.mark.anyio

BASE_URL = "https://router.test/api/v1"
SECRET = "secret transcript about the acquisition"
DIM = 4


class FakeEmbeddings:
    """OpenRouter's /embeddings: answers every request with one vector per input (first value is
    the text's length), returned out of order to check `index` is honoured, or with `outcome`."""

    def __init__(self, outcome: httpx.Response | Exception | None = None, *, dim: int = DIM):
        self.outcome = outcome
        self.dim = dim
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if isinstance(self.outcome, httpx.Response):
            return self.outcome
        texts = json.loads(request.content)["input"]
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": [float(len(t))] + [1.0] * (self.dim - 1),
            }
            for i, t in enumerate(texts)
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "google/gemini-embedding-001",
                "data": list(reversed(data)),
                "usage": {"prompt_tokens": 7, "total_tokens": 7, "cost": 1e-6},
            },
        )

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def router_embedder(
    fake: FakeEmbeddings, model: str = "google/gemini-embedding-001", **kwargs
) -> OpenRouterEmbedder:
    return OpenRouterEmbedder(
        model,
        dim=DIM,
        api_key="test-key",
        base_url=BASE_URL,
        transport=httpx.MockTransport(fake.handle),
        **kwargs,
    )


def failure(code: int, message: str = "boom") -> httpx.Response:
    return httpx.Response(code, json={"error": {"code": code, "message": message}})


# OpenRouterEmbedder


async def test_request_asks_for_the_model_at_the_configured_dimensions_without_data_retention():
    fake = FakeEmbeddings()

    await router_embedder(fake).embed(["one", "three"])

    (request,) = fake.requests
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/embeddings"
    assert request.headers["authorization"] == "Bearer test-key"
    body = fake.bodies[0]
    assert body["model"] == "google/gemini-embedding-001"
    assert body["input"] == ["one", "three"]
    assert body["dimensions"] == DIM
    assert body["provider"]["data_collection"] == "deny"
    assert body["provider"]["require_parameters"] is True


async def test_vectors_are_recorded_under_the_gemini_model_name():
    out = await router_embedder(FakeEmbeddings()).embed(["one", "three"])

    assert out == Embeddings(model="gemini-embedding-001", vectors=[[3, 1, 1, 1], [5, 1, 1, 1]])


async def test_batches_keep_the_order_of_the_texts():
    fake = FakeEmbeddings()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    out = await router_embedder(fake, batch_size=2).embed(texts)

    assert [b["input"] for b in fake.bodies] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert [v[0] for v in out.vectors] == [1, 2, 3, 4, 5]


async def test_nothing_to_embed_makes_no_request():
    fake = FakeEmbeddings()

    out = await router_embedder(fake).embed([])

    assert out == Embeddings(model="gemini-embedding-001", vectors=[])
    assert fake.requests == []


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503])
async def test_overload_is_out_of_capacity(code):
    fake = FakeEmbeddings(failure(code, "rate limited"))

    with pytest.raises(LLMOutOfCapacity, match=str(code)):
        await router_embedder(fake).embed([SECRET])


async def test_a_timeout_is_out_of_capacity():
    fake = FakeEmbeddings(httpx.ReadTimeout("slow"))

    with pytest.raises(LLMOutOfCapacity):
        await router_embedder(fake).embed([SECRET])


@pytest.mark.parametrize("code", [400, 401, 402, 403])
async def test_a_request_error_is_final(code):
    fake = FakeEmbeddings(failure(code, "bad request"))

    with pytest.raises(LLMError, match="bad request") as info:
        await router_embedder(fake).embed([SECRET])

    assert not isinstance(info.value, LLMOutOfCapacity)


async def test_vectors_of_the_wrong_size_are_an_error():
    fake = FakeEmbeddings(dim=3)

    with pytest.raises(LLMError, match="dimensions") as info:
        await router_embedder(fake).embed(["x"])

    assert not isinstance(info.value, LLMOutOfCapacity)


async def test_a_missing_vector_is_an_error():
    response = httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] * DIM}]})

    with pytest.raises(LLMError, match="2 texts"):
        await router_embedder(FakeEmbeddings(response)).embed(["x", "y"])


@pytest.mark.parametrize(
    "outcome",
    [
        failure(400, f"cannot embed {SECRET}"),
        failure(400, "invalid input: secret transcript about the"),
        failure(429),
        httpx.ReadTimeout("slow"),
        httpx.Response(200, json={"data": []}),
    ],
)
async def test_errors_never_carry_the_input_or_the_key(outcome):
    with pytest.raises(LLMError) as info:
        await router_embedder(FakeEmbeddings(outcome)).embed([SECRET])

    message = str(info.value)
    assert "transcript about" not in message
    assert "test-key" not in message


# FallbackEmbedder: Gemini, then OpenRouter


class FakeGeminiModels:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0

    async def embed_content(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        vectors = [[float(len(t)), 1.0, 1.0, 1.0] for t in kwargs["contents"]]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])


def gemini_embedder(models: FakeGeminiModels) -> GeminiEmbedder:
    return GeminiEmbedder(
        models=["gemini-embedding-001"],
        dim=DIM,
        client=SimpleNamespace(aio=SimpleNamespace(models=models)),
    )


def api_error(code: int, status: str) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "status": status, "message": "x"}})


async def test_gemini_with_every_model_out_of_capacity_raises_out_of_capacity():
    models = FakeGeminiModels(api_error(429, "RESOURCE_EXHAUSTED"))

    with pytest.raises(LLMOutOfCapacity):
        await gemini_embedder(models).embed(["x"])


async def test_chain_uses_gemini_while_it_answers():
    fake = FakeEmbeddings()
    chain = FallbackEmbedder([gemini_embedder(FakeGeminiModels()), router_embedder(fake)])

    out = await chain.embed(["ab"])

    assert out.model == "gemini-embedding-001"
    assert fake.requests == []
    assert chain.dim == DIM


async def test_chain_falls_through_to_openrouter_when_gemini_is_rate_limited():
    fake = FakeEmbeddings()
    chain = FallbackEmbedder(
        [
            gemini_embedder(FakeGeminiModels(api_error(429, "RESOURCE_EXHAUSTED"))),
            router_embedder(fake),
        ]
    )

    out = await chain.embed(["ab", "c"], task="query")

    assert out == Embeddings(model="gemini-embedding-001", vectors=[[2, 1, 1, 1], [1, 1, 1, 1]])
    assert len(fake.requests) == 1


async def test_chain_does_not_fall_through_on_a_gemini_request_error():
    fake = FakeEmbeddings()
    chain = FallbackEmbedder(
        [
            gemini_embedder(FakeGeminiModels(api_error(400, "INVALID_ARGUMENT"))),
            router_embedder(fake),
        ]
    )

    with pytest.raises(LLMError) as info:
        await chain.embed(["x"])

    assert not isinstance(info.value, LLMOutOfCapacity)
    assert fake.requests == []


async def test_chain_with_every_provider_out_of_capacity_is_out_of_capacity():
    chain = FallbackEmbedder(
        [
            gemini_embedder(FakeGeminiModels(api_error(429, "RESOURCE_EXHAUSTED"))),
            router_embedder(FakeEmbeddings(failure(429))),
        ]
    )

    with pytest.raises(LLMOutOfCapacity) as info:
        await chain.embed(["x"])

    assert "Gemini" in str(info.value) and "OpenRouter" in str(info.value)


def test_chain_refuses_providers_of_different_dimensions():
    other = OpenRouterEmbedder("google/gemini-embedding-001", dim=8, api_key="k", base_url=BASE_URL)

    with pytest.raises(ValueError, match="dimensions"):
        FallbackEmbedder([gemini_embedder(FakeGeminiModels()), other])


async def test_memory_indexed_by_one_provider_is_searchable_with_the_other():
    store = InMemoryMemoryStore(dim=DIM)
    chunk = Chunk(id="c-1", team_id="t-1", meeting_id="m-1", kind="transcript", text="refunds")
    by_gemini = await gemini_embedder(FakeGeminiModels()).embed([chunk.text])
    await store.index([chunk], by_gemini)

    query = await router_embedder(FakeEmbeddings()).embed(["refunds"], task="query")
    hits = await store.search("t-1", query.vectors[0], model=query.model)

    assert [h.chunk.id for h in hits] == ["c-1"]


# make_embedder


ENV = (
    "GEMINI_API_KEY",
    "GEMINI_EMBEDDING_MODEL",
    "GEMINI_EMBEDDING_DIM",
    "GEMINI_EMBEDDING_FALLBACK_MODELS",
    "OPENROUTER_API_KEY",
    "OPENROUTER_EMBEDDING_MODEL",
)


@pytest.fixture
def no_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


GEMINI = {"gemini_api_key": "g", "gemini_embedding_model": "gemini-embedding-001"}
ROUTER = {"openrouter_api_key": "r", "openrouter_embedding_model": "google/gemini-embedding-001"}


def test_make_embedder_chains_gemini_then_openrouter(no_env):
    out = make_embedder(settings(**GEMINI, **ROUTER, gemini_embedding_dim=768))

    assert isinstance(out, FallbackEmbedder)
    gemini, router = out.providers
    assert isinstance(gemini, GeminiEmbedder) and isinstance(router, OpenRouterEmbedder)
    assert router.model == "google/gemini-embedding-001"
    assert router.recorded_model == "gemini-embedding-001"
    assert out.dim == router.dim == gemini.dim == 768


def test_make_embedder_uses_openrouter_alone_without_a_gemini_key(no_env):
    out = make_embedder(
        settings(**ROUTER, gemini_embedding_model="gemini-embedding-001", gemini_embedding_dim=768)
    )

    assert isinstance(out, OpenRouterEmbedder)
    assert out.recorded_model == "gemini-embedding-001"
    assert out.dim == 768


def test_make_embedder_uses_openrouter_alone_without_any_gemini_setting(no_env):
    out = make_embedder(settings(**ROUTER, gemini_embedding_dim=768))

    assert isinstance(out, OpenRouterEmbedder)
    assert out.recorded_model == "gemini-embedding-001"


def test_make_embedder_refuses_a_different_model_on_openrouter(no_env):
    with pytest.raises(ValueError, match="OPENROUTER_EMBEDDING_MODEL"):
        make_embedder(
            settings(
                **GEMINI,
                openrouter_api_key="r",
                openrouter_embedding_model="openai/text-embedding-3-small",
                gemini_embedding_dim=768,
            )
        )


def test_make_embedder_keeps_gemini_alone_without_the_openrouter_model(no_env):
    out = make_embedder(settings(**GEMINI, openrouter_api_key="r", gemini_embedding_dim=768))

    assert isinstance(out, GeminiEmbedder)


def test_make_embedder_without_a_dimension_names_it(no_env):
    with pytest.raises(LLMUnavailable, match="GEMINI_EMBEDDING_DIM"):
        make_embedder(settings(**GEMINI, **ROUTER))


def test_make_embedder_with_neither_provider_names_both(no_env):
    with pytest.raises(LLMUnavailable) as info:
        make_embedder(settings(gemini_embedding_dim=768))

    assert "GEMINI_API_KEY" in str(info.value)
    assert "OPENROUTER_EMBEDDING_MODEL" in str(info.value)


# live: OpenRouter only (Gemini's embedding quota is the reason this exists)


live_settings = Settings()


@pytest.mark.live
@pytest.mark.skipif(
    not (live_settings.openrouter_api_key and live_settings.openrouter_embedding_model),
    reason="set OPENROUTER_API_KEY and OPENROUTER_EMBEDDING_MODEL to run",
)
async def test_live_openrouter_embeds_two_texts_at_768_dimensions():
    embedder = OpenRouterEmbedder(
        live_settings.openrouter_embedding_model,
        dim=768,
        api_key=live_settings.openrouter_api_key,
        base_url=live_settings.openrouter_url,
    )

    out = await embedder.embed(["Bob refunds the affected users.", "Who refunds the users?"])
    print(f"model {out.model}; usage {embedder.last_usage}")

    assert out.model == "gemini-embedding-001"
    assert [len(v) for v in out.vectors] == [768, 768]
