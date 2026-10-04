from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class WorkerUnavailable(RuntimeError):
    """A setting the worker needs is missing. Never replaced by a stand-in."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None

    elevenlabs_api_key: str | None = None
    elevenlabs_api_url: str = "https://api.elevenlabs.io"
    elevenlabs_stt_model: str | None = None
    # ISO 639-1 to pin one language; unset lets Scribe detect each utterance's, so speech in
    # another language can be translated into English (#106).
    elevenlabs_stt_language: str | None = None
    # Show and save speech in other languages in English (#106). A sentence still going after
    # translation_provisional_seconds is translated so far, again each period while it grows.
    translate_speech: bool = True
    translation_provisional_seconds: float = Field(default=1.5, gt=0)
    elevenlabs_tts_model: str | None = None
    elevenlabs_voice_id: str | None = None  # the default; a team's chosen voice wins

    brain_url: str | None = None
    brain_internal_token: str | None = None

    # The brain expects its ticks every 30-60 s, never per utterance; it rate-limits the model
    # calls behind them itself.
    agenda_tick_seconds: float = Field(default=30, gt=0)
    fact_check_tick_seconds: float = Field(default=60, gt=0)
    # A spoken answer stops after the last whole sentence within this many characters.
    spoken_answer_max_chars: int = Field(default=600, gt=0)


def missing(settings: Settings) -> list[str]:
    """The settings the worker still needs, by their environment names."""
    needed = {
        "LIVEKIT_URL": settings.livekit_url,
        "LIVEKIT_API_KEY": settings.livekit_api_key,
        "LIVEKIT_API_SECRET": settings.livekit_api_secret,
        "ELEVENLABS_API_KEY": settings.elevenlabs_api_key,
        "ELEVENLABS_STT_MODEL": settings.elevenlabs_stt_model,
        "ELEVENLABS_TTS_MODEL": settings.elevenlabs_tts_model,
        "ELEVENLABS_VOICE_ID": settings.elevenlabs_voice_id,
        "BRAIN_URL": settings.brain_url,
        "BRAIN_INTERNAL_TOKEN": settings.brain_internal_token,
    }
    return [name for name, value in needed.items() if not value]


def require(settings: Settings) -> None:
    """Raises WorkerUnavailable naming every missing setting at once."""
    if gaps := missing(settings):
        raise WorkerUnavailable(f"The realtime worker is not configured: set {', '.join(gaps)}")
