from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Missing credentials mean the matching source is unavailable, never simulated."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str | None = None
    transcript_retention_days: int = Field(default=14, ge=1)

    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_jwt_secret: str | None = None  # legacy HS256 secret; asymmetric keys use the JWKS
    supabase_jwt_audience: str = "authenticated"

    gemini_api_key: str | None = None
    gemini_model: str | None = None
    gemini_fallback_models: str | None = None  # comma-separated, tried in order on overload
    gemini_attempts: int = 2  # per model, including the first try
    gemini_max_delay: float = 4.0  # seconds between retries on the same model
    gemini_embedding_model: str | None = None
    gemini_embedding_dim: int | None = None
    # comma-separated; one call never mixes models, and memory search only compares like with like
    gemini_embedding_fallback_models: str | None = None

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

    brain_internal_token: str | None = None
