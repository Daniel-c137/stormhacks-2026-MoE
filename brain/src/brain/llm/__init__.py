"""Gemini behind one small interface. Tests and offline runs use MockLLM explicitly."""

from brain.config import Settings

from .base import LLM, Embedder, LLMError, LLMUnavailable
from .gemini import GeminiLLM
from .mock import MockLLM

__all__ = ["LLM", "Embedder", "GeminiLLM", "LLMError", "LLMUnavailable", "MockLLM", "make_llm"]


def make_llm(settings: Settings | None = None) -> LLM:
    settings = settings or Settings()
    required = {"GEMINI_API_KEY": settings.gemini_api_key, "GEMINI_MODEL": settings.gemini_model}
    if missing := [name for name, value in required.items() if not value]:
        raise LLMUnavailable(f"Gemini is not configured: set {' and '.join(missing)}")
    models = [settings.gemini_model]
    for name in (settings.gemini_fallback_models or "").split(","):
        if (name := name.strip()) and name not in models:
            models.append(name)
    return GeminiLLM(
        models=models,
        api_key=settings.gemini_api_key,
        attempts=settings.gemini_attempts,
        max_delay=settings.gemini_max_delay,
    )
