from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from .agent import Source

AgendaItemStatus = Literal["pending", "covered", "skipped"]


class AgendaItem(BaseModel):
    id: str
    title: str
    why: str | None = None
    owner_id: str | None = None
    sources: list[Source] = []
    status: AgendaItemStatus = "pending"


class Agenda(BaseModel):
    meeting_id: str
    items: list[AgendaItem]
    generated_at: datetime


class AgendaNudge(BaseModel):
    """A reminder that an agenda item has not come up yet."""

    meeting_id: str
    item_id: str
    text: str
