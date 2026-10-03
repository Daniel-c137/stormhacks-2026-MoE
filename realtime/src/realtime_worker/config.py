from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None

    elevenlabs_api_key: str | None = None
    elevenlabs_stt_model: str | None = None
    elevenlabs_tts_model: str | None = None
    elevenlabs_voice_id: str | None = None

    brain_url: str | None = None
    brain_internal_token: str | None = None
