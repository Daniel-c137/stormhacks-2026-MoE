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
    gemini_embedding_model: str | None = None
    gemini_embedding_dim: int | None = None

    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None
    # Join tokens are only checked on connect, so a short life doesn't affect anyone in the call.
    livekit_token_ttl_seconds: int = 600

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
