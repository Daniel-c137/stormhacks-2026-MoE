"""Gemini behind small interfaces. Tests and offline runs use MockLLM and MockEmbedder
explicitly."""

from brain.config import Settings

from .base import LLM, Embedder, Embeddings, EmbedTask, LLMError, LLMUnavailable
from .gemini import GeminiEmbedder, GeminiLLM
from .mock import MockEmbedder, MockLLM

__all__ = [
    "LLM",
    "EmbedTask",
    "Embedder",
    "Embeddings",
    "GeminiEmbedder",
    "GeminiLLM",
    "LLMError",
    "LLMUnavailable",
    "MockEmbedder",
    "MockLLM",
    "make_embedder",
    "make_llm",
]


def make_llm(settings: Settings | None = None) -> LLM:
    settings = settings or Settings()
    required = {"GEMINI_API_KEY": settings.gemini_api_key, "GEMINI_MODEL": settings.gemini_model}
    if missing := [name for name, value in required.items() if not value]:
        raise LLMUnavailable(f"Gemini is not configured: set {' and '.join(missing)}")
    return GeminiLLM(
        models=model_chain(settings.gemini_model, settings.gemini_fallback_models),
        api_key=settings.gemini_api_key,
        attempts=settings.gemini_attempts,
        max_delay=settings.gemini_max_delay,
    )


def make_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or Settings()
    key, model, dim = (
        settings.gemini_api_key,
        settings.gemini_embedding_model,
        settings.gemini_embedding_dim,
    )
    if not (key and model and dim):
        required = {
            "GEMINI_API_KEY": key,
            "GEMINI_EMBEDDING_MODEL": model,
            "GEMINI_EMBEDDING_DIM": dim,
        }
        missing = [name for name, value in required.items() if not value]
        raise LLMUnavailable(f"Gemini embeddings are not configured: set {', '.join(missing)}")
    return GeminiEmbedder(
        models=model_chain(model, settings.gemini_embedding_fallback_models),
        dim=dim,
        api_key=key,
        attempts=settings.gemini_attempts,
        max_delay=settings.gemini_max_delay,
    )


def model_chain(primary: str, fallbacks: str | None) -> list[str]:
    """The primary model, then each comma-separated fallback once."""
    models = [primary]
    for name in (fallbacks or "").split(","):
        if (name := name.strip()) and name not in models:
            models.append(name)
    return models
