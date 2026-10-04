"""Profile, team, members and workspace settings (board -> brain)."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile

from contracts import ConnectorStatus, Person, ProfileUpdate, Team, TeamSettings, Voice

from ..config import Settings
from ..connectors import connector_statuses
from ..store import NotFound, Store
from ..voices import VoicesFailed, VoicesUnavailable, fetch_voices, with_default
from ..zones import is_zone
from .deps import ADMIN_ONLY, current_user, get_http_transport, get_settings, get_store, user_team

router = APIRouter(tags=["team"])

MAX_NAME_LENGTH = 80
MAX_PHOTO_BYTES = 2 * 1024 * 1024
PHOTO_TYPES = {b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg"}
INTERRUPT_MINUTES = range(1, 16)  # the design's dot selector
MAX_TEXT_SETTING = 200
CONNECTOR_FIELDS = ("github", "jira")
CONNECTOR_STATE = {"connected", "indexed_at", "files"}  # the server's, never set by a person


# profile


@router.get("/me")
async def get_me(user: Person = Depends(current_user), store: Store = Depends(get_store)) -> Person:
    return await own_profile(store, user)


@router.patch("/me")
async def update_me(
    body: ProfileUpdate, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Person:
    """The name shows on tiles, in tokens, transcripts and reports."""
    me = await own_profile(store, user)
    if body.name is None:
        return me
    name = " ".join(body.name.split())
    if not name:
        raise HTTPException(status_code=422, detail="Name is required")
    if len(name) > MAX_NAME_LENGTH:
        raise HTTPException(
            status_code=422, detail=f"Name must be at most {MAX_NAME_LENGTH} characters"
        )
    words = name.split()
    initials = (words[0][0] + (words[-1][0] if len(words) > 1 else "")).upper()
    return await store.update_person(
        me.model_copy(update={"name": name, "short": words[0], "initials": initials})
    )


@router.post("/me/photo")
async def upload_photo(
    file: UploadFile, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Person:
    """Shown on the tile when the camera is off. JPG or PNG, told apart by the bytes."""
    await own_profile(store, user)
    data = await file.read(MAX_PHOTO_BYTES + 1)
    if not data:
        raise HTTPException(status_code=422, detail="The photo is empty")
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(status_code=413, detail="The photo must be 2 MB or smaller")
    content_type = next((t for magic, t in PHOTO_TYPES.items() if data.startswith(magic)), None)
    if content_type is None:
        raise HTTPException(status_code=415, detail="The photo must be a JPG or PNG")
    return await store.save_photo(user.id, content_type, data)


@router.delete("/me/photo")
async def delete_photo(
    user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Person:
    await own_profile(store, user)
    return await store.delete_photo(user.id)


@router.get("/people/{person_id}/photo")
async def get_photo(
    person_id: str, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> Response:
    """Teammates' photos only; anyone else's looks like it does not exist."""
    team = await user_team(store, user)
    if person_id not in team.member_ids:
        raise HTTPException(status_code=404, detail="Photo not found")
    try:
        content_type, data = await store.photo(person_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Photo not found") from None
    return Response(
        content=data,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff"},
    )


async def own_profile(store: Store, user: Person) -> Person:
    try:
        return await store.person(user.id)
    except NotFound:
        raise HTTPException(status_code=404, detail="Profile not found") from None


# team


@router.get("/team")
async def get_team(user: Person = Depends(current_user), store: Store = Depends(get_store)) -> Team:
    return await user_team(store, user)


@router.get("/team/members")
async def list_members(
    user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> list[Person]:
    team = await user_team(store, user)
    return await store.members(team.id)


# workspace settings


@router.get("/settings")
async def read_team_settings(
    user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> TeamSettings:
    team = await user_team(store, user)
    return await store.settings(team.id)


@router.put("/settings")
async def write_team_settings(
    body: TeamSettings, user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> TeamSettings:
    """Any team member may change them, but only an admin the connectors. Always the caller's own
    team; connection and index state belong to the server and are kept as they are."""
    if body.interrupt_minutes not in INTERRUPT_MINUTES:
        raise HTTPException(
            status_code=422,
            detail=f"interrupt_minutes must be {INTERRUPT_MINUTES[0]} to {INTERRUPT_MINUTES[-1]}",
        )
    timezone = body.timezone.strip()
    if not is_zone(timezone):
        raise HTTPException(
            status_code=422,
            detail="timezone must be an IANA time zone name, e.g. America/Vancouver or UTC",
        )
    team = await user_team(store, user)
    current = await store.settings(team.id)
    if not user.is_admin and connectors_changed(current, body):
        raise HTTPException(status_code=403, detail=ADMIN_ONLY)
    github = current.github.model_copy(
        update={"repo": text(body.github.repo), "ref": text(body.github.ref)}
    )
    jira = current.jira.model_copy(
        update={"site": text(body.jira.site), "project": text(body.jira.project)}
    )
    return await store.save_settings(
        body.model_copy(
            update={
                "team_id": team.id,
                "github": github,
                "jira": jira,
                "voice": text(body.voice),
                "wake_phrase": text(body.wake_phrase),
                "timezone": timezone,
            }
        )
    )


@router.get("/settings/connectors")
async def list_connectors(
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    config: Settings = Depends(get_settings),
) -> list[ConnectorStatus]:
    """GitHub and Jira, each connected, not configured or failing, checked live."""
    team = await user_team(store, user)
    return await connector_statuses(config, await store.settings(team.id))


@router.get("/voices")
async def list_voices(
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    config: Settings = Depends(get_settings),
    transport: httpx.AsyncBaseTransport | None = Depends(get_http_transport),
) -> list[Voice]:
    """The account's voices, the agent's default (ELEVENLABS_VOICE_ID) first and labelled."""
    await user_team(store, user)
    try:
        voices = await fetch_voices(config, transport=transport)
    except VoicesUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    except VoicesFailed as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    return with_default(voices, config.elevenlabs_voice_id)


def connectors_changed(current: TeamSettings, body: TeamSettings) -> bool:
    """Whether the body changes a connector value a person edits (today the repository and ref,
    and the Jira site and project), compared as saved; the server's state is ignored."""
    return any(editable(body, name) != editable(current, name) for name in CONNECTOR_FIELDS)


def editable(settings: TeamSettings, connector: str) -> dict:
    values = getattr(settings, connector).model_dump(exclude=CONNECTOR_STATE)
    return {key: text(v) if isinstance(v, str) else v for key, v in values.items()}


def text(value: str | None) -> str | None:
    """Trimmed; blank means unset."""
    if value is None or not (value := value.strip()):
        return None
    if len(value) > MAX_TEXT_SETTING:
        raise HTTPException(
            status_code=422, detail=f"Settings text must be at most {MAX_TEXT_SETTING} characters"
        )
    return value
