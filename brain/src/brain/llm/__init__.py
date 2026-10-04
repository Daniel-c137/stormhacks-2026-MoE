"""Gemini, with OpenRouter as the fallback when it is out of capacity, behind small interfaces.
Embeddings are Gemini only. Tests and offline runs use MockLLM and MockEmbedder explicitly."""

from brain.config import Settings

from .base import (
    LLM,
    Embedder,
    Embeddings,
    EmbedTask,
    LLMError,
    LLMOutOfCapacity,
    LLMUnavailable,
)
from .chain import FallbackLLM
from .gemini import GeminiEmbedder, GeminiLLM
from .mock import MockEmbedder, MockLLM
from .openrouter import OpenRouterLLM

__all__ = [
    "LLM",
    "EmbedTask",
    "Embedder",
    "Embeddings",
    "FallbackLLM",
    "GeminiEmbedder",
    "GeminiLLM",
    "LLMError",
    "LLMOutOfCapacity",
    "LLMUnavailable",
    "MockEmbedder",
    "MockLLM",
    "OpenRouterLLM",
    "comma_list",
    "make_embedder",
    "make_llm",
]


def make_llm(settings: Settings | None = None) -> LLM:
    """Gemini when GEMINI_API_KEY and GEMINI_MODEL are set, then OpenRouter when
    OPENROUTER_API_KEY and OPENROUTER_MODELS are; one provider alone is used directly."""
    settings = settings or Settings()
    providers: list[LLM] = []
    if settings.gemini_api_key and settings.gemini_model:
        providers.append(
            GeminiLLM(
                models=model_chain(settings.gemini_model, settings.gemini_fallback_models),
                api_key=settings.gemini_api_key,
                attempts=settings.gemini_attempts,
                max_delay=settings.gemini_max_delay,
            )
        )
    openrouter_models = comma_list(settings.openrouter_models)
    if settings.openrouter_api_key and openrouter_models:
        providers.append(
            OpenRouterLLM(
                openrouter_models,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_url,
            )
        )
    if not providers:
        raise LLMUnavailable(
            "Gemini is not configured (set GEMINI_API_KEY and GEMINI_MODEL), and neither is "
            "the OpenRouter fallback (set OPENROUTER_API_KEY and OPENROUTER_MODELS)"
        )
    return providers[0] if len(providers) == 1 else FallbackLLM(providers)


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
    return comma_list(f"{primary},{fallbacks or ''}")


def comma_list(value: str | None) -> list[str]:
    """Each comma-separated name once, in order, without blanks."""
    names: list[str] = []
    for name in (value or "").split(","):
        if (name := name.strip()) and name not in names:
            names.append(name)
    return names
