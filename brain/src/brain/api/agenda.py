"""The lobby agenda (board -> brain): the team edits a timeboxed list before and during the
meeting; the agent only rewrites topics and suggests items, and never saves either."""

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from contracts import (
    Agenda,
    AgendaItem,
    AgendaRewriteRequest,
    AgendaRewriteResponse,
    AgendaSuggestions,
    AgendaUpdate,
    Person,
)

from ..agent.agenda import (
    MINUTES_MAX,
    MINUTES_MIN,
    TITLE_MAX,
    TOPIC_MAX,
    jira_inputs,
    rewrite_topic,
    store_inputs,
    suggest_items,
    valid_minutes,
)
from ..config import Settings
from ..jira import JiraError, JiraReader, JiraUnavailable, jira_config
from ..llm import LLM, LLMError
from ..store import Store
from .deps import current_user, get_llm, get_settings, get_store, team_meeting

router = APIRouter(tags=["agenda"])

EDITABLE = {"scheduled", "live"}
GITHUB_NOT_READ = "GitHub: open issues and pull requests are not read for suggestions yet"


@router.get("/meetings/{meeting_id}/agenda")
async def get_agenda(
    meeting_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Agenda:
    """The saved agenda, or an empty one."""
    meeting = await team_meeting(store, user, meeting_id)
    saved = await store.agenda(meeting.id)
    return saved or Agenda(meeting_id=meeting.id, items=[], generated_at=datetime.now(UTC))


@router.put("/meetings/{meeting_id}/agenda")
async def update_agenda(
    meeting_id: str,
    body: AgendaUpdate,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
) -> Agenda:
    """Saves the whole edited list in order. Any teammate, until the meeting ends."""
    meeting = await team_meeting(store, user, meeting_id)
    if meeting.status not in EDITABLE:
        raise HTTPException(status_code=409, detail="The meeting has ended; its agenda is final")

    ids = [item.id for item in body.items if item.id]
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=422, detail="An agenda item appears more than once")
    for item in body.items:
        title = item.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="Agenda items need a title")
        if len(title) > TITLE_MAX:
            raise HTTPException(
                status_code=422, detail=f"Agenda titles are at most {TITLE_MAX} characters"
            )
        if not valid_minutes(item.minutes):
            raise HTTPException(
                status_code=422,
                detail=f"A timebox is {MINUTES_MIN} to {MINUTES_MAX} minutes, or none",
            )

    now = datetime.now(UTC)
    saved = await store.agenda(meeting.id)
    existing = {item.id: item for item in saved.items} if saved else {}
    items: list[AgendaItem] = []
    for edit in body.items:
        changes = {"title": edit.title.strip(), "minutes": edit.minutes}
        if edit.id and edit.id in existing:
            items.append(existing[edit.id].model_copy(update=changes))
        else:
            items.append(AgendaItem(id=uuid4().hex, added_by=user.id, **changes))
    agenda = Agenda(
        meeting_id=meeting.id,
        items=items,
        generated_at=saved.generated_at if saved else now,
        updated_at=now,
    )
    return await store.save_agenda(agenda)


@router.post("/agenda/rewrite")
async def rewrite_agenda_item(
    body: AgendaRewriteRequest, user: Person = Depends(current_user), llm: LLM = Depends(get_llm)
) -> AgendaRewriteResponse:
    """One clear agenda item for a rough topic. A suggestion only; nothing is saved."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Type a topic to rewrite")
    if len(body.text) > TOPIC_MAX:
        raise HTTPException(status_code=422, detail=f"Topics are at most {TOPIC_MAX} characters")
    try:
        return AgendaRewriteResponse(text=await rewrite_topic(llm, body.text))
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Could not rewrite the topic: {e}") from e


@router.post("/meetings/{meeting_id}/agenda/suggest")
async def suggest_agenda(
    meeting_id: str,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_settings),
    llm: LLM = Depends(get_llm),
) -> AgendaSuggestions:
    """Proposed items from the team's earlier reports, open and overdue tasks and unfinished
    Jira work, each with why and its sources. Nothing is saved."""
    meeting = await team_meeting(store, user, meeting_id)
    today = datetime.now(UTC).date()
    inputs = await store_inputs(store, meeting.team_id, meeting.id, today)
    unavailable = [GITHUB_NOT_READ]
    try:
        issues = await JiraReader(jira_config(settings)).unfinished()
    except JiraUnavailable as e:
        unavailable.append(str(e))
    except JiraError as e:
        unavailable.append(f"Jira search failed: {e}")
    else:
        inputs += jira_inputs(issues)
    try:
        items = await suggest_items(llm, meeting, inputs, today)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Could not suggest items: {e}") from e
    return AgendaSuggestions(items=items, unavailable=unavailable)
