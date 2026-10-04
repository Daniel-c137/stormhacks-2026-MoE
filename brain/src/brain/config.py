from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Missing credentials mean the matching source is unavailable, never simulated."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str | None = None  # our Postgres 16 + pgvector, with db/migrations applied
    transcript_retention_days: int = Field(default=14, ge=1)

    # HS256 key for the session tokens the brain issues at login; under 32 characters is unset
    auth_secret: str | None = None
    session_hours: int = Field(default=12, ge=1)
    # Google sign-in needs all three: an OAuth client from Google Cloud ("Web application"), and
    # the public callback URL Google sends the browser to, exactly as registered with Google:
    # https://<domain>/api/auth/google/callback in production. The brain can't build it itself
    # (behind the /api proxy it sees its internal address); its scheme also decides whether the
    # sign-in cookie is Secure.
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_url: str | None = None
    # Where the board is served, for the Google callback to send the browser back to its /login.
    # Unset, it's /login on GOOGLE_REDIRECT_URL's site: right when one origin serves both (Caddy
    # at /api in production, the board's /api rewrite locally). Set it only when they differ.
    board_url: str | None = None

    gemini_api_key: str | None = None
    gemini_model: str | None = None
    gemini_fallback_models: str | None = None  # comma-separated, tried in order on overload or 404
    gemini_attempts: int = 2  # per model, including the first try
    gemini_max_delay: float = 4.0  # seconds between retries on the same model
    # Live translation (#106): its own (cheaper) Gemini model, GEMINI_MODEL when unset, tried
    # once with no fallback, and given up on before the worker's 4 s wait runs out.
    translation_model: str | None = None
    translation_timeout_seconds: float = 3.5
    gemini_embedding_model: str | None = None
    gemini_embedding_dim: int | None = None
    # comma-separated; one call never mixes models, and memory search only compares like with like
    gemini_embedding_fallback_models: str | None = None
    # The fallback when every Gemini model is out of capacity: comma-separated OpenRouter model
    # ids, tried in order on 404, 408, 429 or 5xx
    openrouter_api_key: str | None = None
    openrouter_models: str | None = None
    openrouter_url: str = "https://openrouter.ai/api/v1"
    # The embeddings fallback, e.g. google/gemini-embedding-001: the same model as
    # GEMINI_EMBEDDING_MODEL (its vectors are compared with Gemini's), at GEMINI_EMBEDDING_DIM
    openrouter_embedding_model: str | None = None

    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None
    # Join tokens are only checked on connect, so a short life doesn't affect anyone in the call.
    livekit_token_ttl_seconds: int = 600

    github_mcp_url: str | None = None
    # GitLab's MCP server, https://<instance>/api/v4/mcp (links to code use that instance), or
    # the world mock
    gitlab_mcp_url: str | None = None
    jira_mcp_url: str | None = None
    github_token: str | None = None
    github_repo: str | None = None
    jira_base_url: str | None = None
    jira_cloud_id: str | None = None
    jira_project_key: str | None = None
    connector_timeout: float = 5.0  # seconds to reach an MCP server and list its tools

    elevenlabs_api_key: str | None = None
    elevenlabs_api_url: str = "https://api.elevenlabs.io"
    # the agent's voice and speech model, shared with realtime's spoken answers; a team's chosen
    # voice wins over the default voice
    elevenlabs_voice_id: str | None = None
    elevenlabs_tts_model: str | None = None

    brain_internal_token: str | None = None

    # seconds the write-up waits after the host ends, so the worker's last final segments land
    pipeline_settle_seconds: float = 8.0
    # minutes without progress after which a write-up nobody here is running may be retried
    pipeline_stale_minutes: float = Field(default=10.0, gt=0)
