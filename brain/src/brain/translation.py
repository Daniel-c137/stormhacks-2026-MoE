"""Live translation of non-English speech into English for captions and the transcript (#106).

The worker calls this per non-English utterance (and on a still-open sentence every 1.5 s), so
it is one small structured call: detect the language and give the English. Nothing here reasons
about the meeting.
"""

from pydantic import BaseModel, Field

from contracts import TranslateResponse, get_identity
from contracts.language import normalise_language

from .llm import LLM


class Translation(BaseModel):
    """What the model is asked for."""

    language: str = Field(description="ISO 639-1 code of the speech, lowercase, e.g. 'es'.")
    english: str = Field(description="The speech in English; unchanged if it already is.")


def system_prompt() -> str:
    agent = get_identity().agent_name
    return f"""You translate live meeting speech for a software team's captions.

Say which language the speech is in, as an ISO 639-1 code, and give it in English.
- Translate faithfully and plainly. Don't summarise, answer or add anything.
- Keep names (people, products, and the meeting assistant "{agent}"), numbers and identifiers
  such as issue keys (DS-104), pull request numbers, versions and code exactly as said.
- The speech may stop mid-sentence: translate what is there, without finishing it.
- If it is already English, give it back unchanged."""


class TranslationFailed(RuntimeError):
    """The model answered with something unusable."""


async def translate(llm: LLM, text: str, language: str | None = None) -> TranslateResponse:
    """`language` is Scribe's detected code when it gave one; it is trusted over the model's.
    English comes back word for word, never as the model's rewording."""
    hint = normalise_language(language)
    prompt = f"Language detected by the transcriber: {hint or 'unknown'}\n\nSpeech:\n{text}"
    answer = await llm.generate_structured(prompt, Translation, system=system_prompt())
    detected = hint or normalise_language(answer.language)
    if detected is None:
        raise TranslationFailed(f"the model named the language {answer.language!r}")
    if detected == "en":
        return TranslateResponse(language="en", text=text)
    english = answer.english.strip()
    if not english:
        raise TranslationFailed("the model returned no English")
    return TranslateResponse(language=detected, text=english)
