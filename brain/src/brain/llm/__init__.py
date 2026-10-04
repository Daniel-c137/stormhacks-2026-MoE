"""Gemini, with OpenRouter as the fallback when it is out of capacity, behind small interfaces.
Embeddings fall back only to the same model on OpenRouter, so stored vectors stay comparable.
Tests and offline runs use MockLLM and MockEmbedder explicitly."""

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
from .chain import FallbackEmbedder, FallbackLLM
from .gemini import GeminiEmbedder, GeminiLLM
from .mock import MockEmbedder, MockLLM
from .openrouter import OpenRouterEmbedder, OpenRouterLLM, recorded_model_name

__all__ = [
    "LLM",
    "EmbedTask",
    "Embedder",
    "Embeddings",
    "FallbackEmbedder",
    "FallbackLLM",
    "GeminiEmbedder",
    "GeminiLLM",
    "LLMError",
    "LLMOutOfCapacity",
    "LLMUnavailable",
    "MockEmbedder",
    "MockLLM",
    "OpenRouterEmbedder",
    "OpenRouterLLM",
    "comma_list",
    "make_embedder",
    "make_llm",
    "make_translation_llm",
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


def make_translation_llm(settings: Settings | None = None) -> LLM:
    """Live translation (#106) runs on every non-English utterance, so it must stay cheap and
    fast: TRANSLATION_MODEL (or GEMINI_MODEL) with one attempt and no fallback chain. A Gemini
    quota error fails the caption instead of moving to paid OpenRouter; OpenRouter is used only
    when Gemini isn't configured at all."""
    settings = settings or Settings()
    model = settings.translation_model or settings.gemini_model
    if settings.gemini_api_key and model:
        return GeminiLLM(models=(model,), api_key=settings.gemini_api_key, attempts=1, max_delay=0)
    return make_llm(settings)


def make_embedder(settings: Settings | None = None) -> Embedder:
    """Gemini when GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL are set, then the same model on
    OpenRouter when OPENROUTER_API_KEY and OPENROUTER_EMBEDDING_MODEL are; one alone is used
    directly. Both make GEMINI_EMBEDDING_DIM-dimension vectors. A different model on OpenRouter
    is refused (ValueError): its vectors could not be compared with Gemini's."""
    settings = settings or Settings()
    key, model, dim = (
        settings.gemini_api_key,
        settings.gemini_embedding_model,
        settings.gemini_embedding_dim,
    )
    router_key, router_model = settings.openrouter_api_key, settings.openrouter_embedding_model
    if router_key and router_model and model and recorded_model_name(router_model) != model:
        raise ValueError(
            f"OPENROUTER_EMBEDDING_MODEL ({router_model}) is not GEMINI_EMBEDDING_MODEL "
            f"({model}) on OpenRouter (google/{model}); vectors from different models cannot "
            "be compared"
        )
    gemini_ready = bool(key and model)
    router_ready = bool(router_key and router_model)
    if not (dim and (gemini_ready or router_ready)):
        required = {
            "GEMINI_EMBEDDING_DIM": dim,
            "GEMINI_API_KEY": key,
            "GEMINI_EMBEDDING_MODEL": model,
        }
        missing = [name for name, value in required.items() if not value]
        raise LLMUnavailable(
            f"Embeddings are not configured: set {', '.join(missing)} (or OPENROUTER_API_KEY "
            "and OPENROUTER_EMBEDDING_MODEL for the same model on OpenRouter)"
        )
    providers: list[Embedder] = []
    if key and model:
        providers.append(
            GeminiEmbedder(
                models=model_chain(model, settings.gemini_embedding_fallback_models),
                dim=dim,
                api_key=key,
                attempts=settings.gemini_attempts,
                max_delay=settings.gemini_max_delay,
            )
        )
    if router_key and router_model:
        providers.append(
            OpenRouterEmbedder(
                router_model, dim=dim, api_key=router_key, base_url=settings.openrouter_url
            )
        )
    return providers[0] if len(providers) == 1 else FallbackEmbedder(providers)


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
