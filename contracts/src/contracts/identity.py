"""Working product and agent names. Rename them in contracts/identity.json only."""

from functools import cache
from pathlib import Path

from pydantic import BaseModel

IDENTITY_FILE = Path(__file__).resolve().parents[2] / "identity.json"

# Stable LiveKit identity for the agent. Never derived from the display name.
AGENT_PARTICIPANT_ID = "agent"


class Identity(BaseModel):
    product_name: str
    agent_name: str

    @property
    def wake_phrase(self) -> str:
        """The agent's name alone: "Polaris, what's blocking DS-104?"."""
        return self.agent_name

    @property
    def mention(self) -> str:
        return f"@{self.agent_name}"


@cache
def get_identity() -> Identity:
    return Identity.model_validate_json(IDENTITY_FILE.read_text())
