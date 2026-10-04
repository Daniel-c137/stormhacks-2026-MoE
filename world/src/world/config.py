from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

WORLD_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WORLD_ROOT.parent
SNAPSHOTS_DIR = WORLD_ROOT / "snapshots"
MEETINGS_DIR = WORLD_ROOT / "meetings"
OVERLAY_DIR = WORLD_ROOT / ".overlay"
MOCK_DATA_DIR = REPO_ROOT / "mock-data"  # records in the real APIs' shapes, e.g. jira/issues.json

SnapshotName = Literal["dev", "demo"]


class SnapshotSpec(BaseModel):
    data_until: date
    today: datetime  # the agent's "today" for this snapshot


class WorldSpec(BaseModel):
    company: str
    github_repo: str
    jira_project: str
    snapshots: dict[SnapshotName, SnapshotSpec]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    world_snapshot: SnapshotName = "dev"
    world_today_override: datetime | None = None
    world_github_mcp_port: int = 8101
    world_jira_mcp_port: int = 8102
    world_overlay_dir: Path = OVERLAY_DIR


def world_spec() -> WorldSpec:
    return WorldSpec.model_validate_json((WORLD_ROOT / "world.json").read_text())
