from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Missing credentials mean the matching source is unavailable, never simulated."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    supabase_url: str | None = None
    supabase_service_role_key: str | None = None

    gemini_api_key: str | None = None
    gemini_model: str | None = None
    gemini_fallback_models: str | None = None  # comma-separated, tried in order on overload
    gemini_attempts: int = 2  # per model, including the first try
    gemini_max_delay: float = 4.0  # seconds between retries on the same model
    gemini_embedding_model: str | None = None
    gemini_embedding_dim: int | None = None

    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None

    github_mcp_url: str | None = None
    jira_mcp_url: str | None = None
    github_token: str | None = None
    github_repo: str | None = None
    jira_base_url: str | None = None
    jira_cloud_id: str | None = None
    jira_email: str | None = None
    jira_api_token: str | None = None
    jira_project_key: str | None = None
    connector_timeout: float = 5.0  # seconds to reach an MCP server and list its tools

    elevenlabs_api_key: str | None = None
    elevenlabs_api_url: str = "https://api.elevenlabs.io"

    brain_internal_token: str | None = None
