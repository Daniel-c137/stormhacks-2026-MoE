from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .agent import Source

AgendaItemStatus = Literal["pending", "covered", "skipped"]


class AgendaItem(BaseModel):
    id: str
    title: str
    why: str | None = None
    owner_id: str | None = None
    sources: list[Source] = []
    status: AgendaItemStatus = "pending"
    minutes: int | None = None  # timebox
    added_by: str | None = None  # person id; None when the agent proposed it
    # Timekeeping during the meeting; times are seconds from the meeting start.
    discussed_s: float = Field(default=0, ge=0)  # talk time attributed to this item so far
    nudged_t: float | None = None  # when the agent nudged that it had not come up; once at most
    # Who marked it covered: a person id, or AGENT_PARTICIPANT_ID when the tracker did. None
    # while it is not covered. covered_t is when; None if the meeting had not started.
    covered_by: str | None = None
    covered_t: float | None = None


class Agenda(BaseModel):
    meeting_id: str
    items: list[AgendaItem]
    generated_at: datetime
    updated_at: datetime | None = None  # last human edit
    current_item_id: str | None = None  # being discussed now; None when off the agenda
    tracked_until: float | None = None  # transcript seconds tracked so far; None before any
    revision: int = Field(default=0, ge=0)  # bumped by every save; 0 until first saved


class AgendaSuggestions(BaseModel):
    """Proposed items, not saved; a person adds the ones they want."""

    items: list[AgendaItem]
    unavailable: list[str] = []  # sources that were missing or failed; never papered over


class AgendaNudge(BaseModel):
    """A reminder that an agenda item has not come up yet."""

    meeting_id: str
    item_id: str
    text: str
