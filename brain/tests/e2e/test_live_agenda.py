"""Live: a rough lobby topic rewritten by real Gemini. Needs GEMINI_API_KEY and GEMINI_MODEL
(optionally GEMINI_FALLBACK_MODELS). Deselected unless pytest runs with `-m live`."""

import pytest

from brain.agent.agenda import rewrite_topic
from brain.config import Settings
from brain.llm import GeminiLLM, make_llm

settings = Settings()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.gemini_api_key and settings.gemini_model),
        reason="set GEMINI_API_KEY and GEMINI_MODEL to run against Gemini",
    ),
]


@pytest.mark.anyio
async def test_live_gemini_rewrites_redis_vs_postgres_into_one_agenda_item():
    llm = make_llm(settings)
    assert isinstance(llm, GeminiLLM)

    title = await rewrite_topic(llm, "redis vs postgres?")
    print(f"{title!r} (answered by {llm.last_model})")

    assert title and "\n" not in title
    assert len(title) <= 80 and len(title.split()) <= 12
    assert any(word in title.lower() for word in ("queue", "redis", "postgres"))
