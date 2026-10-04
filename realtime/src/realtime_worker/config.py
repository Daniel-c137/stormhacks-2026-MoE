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
    # ISO 639-1 language Scribe is pinned to. A meeting with live translation on (#106) lets
    # Scribe detect each utterance's language instead.
    elevenlabs_stt_language: str = "en"
    # Seconds of silence after which Scribe ends an utterance and sends its final. Scribe's own
    # (about 1.5 s) made Polaris start working 1.8 s after a question ended; 1.0 makes it 1.3 s.
    # Shorter still splits a sentence with a pause in it into more captions. Scribe refuses a
    # session outside 0.3 to 3.0.
    elevenlabs_vad_silence_seconds: float = Field(default=1.0, ge=0.3, le=3.0)
    # With translation on, a sentence still going after this long is translated so far, again
    # each period while it grows.
    translation_provisional_seconds: float = Field(default=1.5, gt=0)
    # After a failed translation (the model down or slow), provisional ones pause this long.
    translation_pause_seconds: float = Field(default=45, ge=0)
    elevenlabs_tts_model: str | None = None
    elevenlabs_voice_id: str | None = None  # the default; a team's chosen voice wins

    brain_url: str | None = None
    brain_internal_token: str | None = None

    # Timer ticks, never per utterance; the brain rate-limits the model calls behind them itself.
    # The agenda's is short so a finished item is ticked within seconds: the brain asks its model
    # only once a tick's new stretch holds enough talk.
    agenda_tick_seconds: float = Field(default=10, gt=0)
    # The brain's JEV_MODEL: with Jev keeping time the agenda is also checked after every caption,
    # this long after it ends, once it has settled in the brain (at least the brain's JEV_SETTLE_S).
    jev_model: str | None = None
    agenda_check_delay_seconds: float = Field(default=2.5, gt=0)
    fact_check_tick_seconds: float = Field(default=60, gt=0)
    # A spoken answer stops after the last whole sentence within this many characters.
    spoken_answer_max_chars: int = Field(default=600, gt=0)

    @property
    def agenda_after_captions(self) -> bool:
        return bool(self.jev_model)


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
