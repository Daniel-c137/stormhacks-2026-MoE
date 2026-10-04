from datetime import datetime

from pydantic import BaseModel

from .agent import Visibility


class ChatMessage(BaseModel):
    """Public messages are saved with the meeting. Private ones are ephemeral and never stored.

    A private message has visibility "private" and its recipient_id; a private question to the
    agent uses the agent's id (AGENT_PARTICIPANT_ID) as recipient_id.
    """

    id: str
    meeting_id: str
    sender_id: str
    sender_name: str
    is_agent: bool
    text: str
    ts: datetime
    visibility: Visibility = "public"
    recipient_id: str | None = None
    snippet_id: str | None = None
