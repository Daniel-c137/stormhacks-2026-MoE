"""Live: one small structured call through OpenRouter, the fallback after Gemini. Needs
OPENROUTER_API_KEY and OPENROUTER_MODELS; only the first model is asked. Gemini is left out of the
settings, so the Gemini key cannot answer. Run it as
`OPENROUTER_MODELS=google/gemini-3.5-flash-lite uv run pytest brain -m live -k openrouter -s`.
Deselected unless pytest runs with `-m live`."""

import pytest

from brain.agent.agenda import rewrite_topic
from brain.config import Settings
from brain.llm import comma_list, make_llm
from brain.llm.openrouter import OpenRouterLLM

configured = Settings()
model = next(iter(comma_list(configured.openrouter_models)), None)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (configured.openrouter_api_key and model),
        reason="set OPENROUTER_API_KEY and OPENROUTER_MODELS to run against OpenRouter",
    ),
]


@pytest.mark.anyio
async def test_live_openrouter_rewrites_a_topic_with_structured_output():
    settings = configured.model_copy(
        update={"gemini_api_key": None, "gemini_model": None, "openrouter_models": model}
    )
    llm = make_llm(settings)
    assert isinstance(llm, OpenRouterLLM)

    title = await rewrite_topic(llm, "redis vs postgres?")
    print(f"{title!r} (answered by {llm.last_model}; usage {llm.last_usage})")

    assert llm.last_model == f"openrouter:{model}"
    assert title and "\n" not in title
    assert any(word in title.lower() for word in ("queue", "redis", "postgres"))
