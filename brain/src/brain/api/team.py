"""Profile, team, members and workspace settings (board -> brain)."""

import re
from collections.abc import Callable

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile

from contracts import (
    CodeRepo,
    CodeRepoChoice,
    ConnectorStatus,
    ConnectorsUpdate,
    Person,
    ProfileUpdate,
    Team,
    TeamSettings,
    Voice,
)

from ..accounts import signs_in
from ..config import Settings
from ..connectors import connector_statuses
from ..gitlab import valid_project
from ..jira import site_host
from ..store import NotFound, Store
from ..voices import VoicesFailed, VoicesUnavailable, fetch_voices, with_default
from ..zones import is_zone
from .deps import (
    current_user,
    get_http_transport,
    get_settings,
    get_store,
    require_admin,
    user_team,
)

router = APIRouter(tags=["team"])

MAX_NAME_LENGTH = 80
MAX_PHOTO_BYTES = 2 * 1024 * 1024
PHOTO_TYPES = {b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg"}
INTERRUPT_MINUTES = range(1, 16)  # the design's dot selector
MAX_TEXT_SETTING = 200


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
    """The team's people; `invited` marks those an admin invited who haven't signed up yet (an
    email and no login)."""
    team = await user_team(store, user)
    return [
        p.model_copy(update={"invited": bool(p.email) and not await signs_in(store, p.id)})
        for p in await store.members(team.id)
    ]


# workspace settings


@router.get("/settings")
async def read_team_settings(
    user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> TeamSettings:
    team = await user_team(store, user)
    return await store.settings(team.id)


@router.put("/settings")
async def write_team_settings(
    body: TeamSettings, user: Person = Depends(require_admin), store: Store = Depends(get_store)
) -> TeamSettings:
    """Admin only. Always the caller's own team. The connectors are kept as saved, whatever the
    body says: they change only at PUT /settings/connectors."""
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
    return await store.save_settings(
        body.model_copy(
            update={
                "team_id": team.id,
                "github": current.github,
                "gitlab": current.gitlab,
                "jira": current.jira,
                "voice": text(body.voice),
                "wake_phrase": text(body.wake_phrase),
                "timezone": timezone,
            }
        )
    )


@router.put("/settings/connectors")
async def write_connectors(
    body: ConnectorsUpdate, user: Person = Depends(require_admin), store: Store = Depends(get_store)
) -> TeamSettings:
    """The team's GitHub repositories, GitLab projects and Jira site and project, replacing
    the saved ones; admins only. A repository that stays keeps its connection and index state
    (a new branch or tag drops its index). Paths are checked and repeats refused."""
    team = await user_team(store, user)
    current = await store.settings(team.id)
    github = repos(body.github, current.github.repos, "GitHub repository", github_path)
    gitlab = repos(body.gitlab, current.gitlab.projects, "GitLab project", gitlab_path)
    jira = current.jira.model_copy(
        update={"site": site_host(text(body.jira.site)), "project": jira_key(body.jira.project)}
    )
    return await store.save_settings(
        current.model_copy(
            update={
                "github": current.github.model_copy(update={"repos": github}),
                "gitlab": current.gitlab.model_copy(update={"projects": gitlab}),
                "jira": jira,
            }
        )
    )


def repos(
    chosen: list[CodeRepoChoice], saved: list[CodeRepo], noun: str, clean: Callable[[str], str]
) -> list[CodeRepo]:
    by_path = {repo.path.casefold(): repo for repo in saved}
    found: list[CodeRepo] = []
    for choice in chosen:
        try:
            path = clean(choice.path)
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"{choice.path.strip()!r} is not a {noun} path"
            ) from None
        if any(repo.path.casefold() == path.casefold() for repo in found):
            raise HTTPException(status_code=422, detail=f"{path} is listed twice")
        ref = text(choice.ref)
        kept = by_path.get(path.casefold())
        if kept is not None and kept.ref == ref:
            found.append(kept.model_copy(update={"path": path}))
        else:
            found.append(CodeRepo(path=path, ref=ref))
    return found


GITHUB_PATH = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}")


def github_path(path: str) -> str:
    """owner/name, as GitHub allows them; a pasted github.com link is accepted."""
    clean = re.sub(r"^(?:https?://)?(?:www\.)?github\.com/", "", path.strip()).strip("/")
    clean = clean.removesuffix(".git")
    if not GITHUB_PATH.fullmatch(clean) or clean.split("/")[1] in (".", ".."):
        raise ValueError(path)
    return clean


def gitlab_path(path: str) -> str:
    """group/project, subgroups included; a pasted link to the project is accepted."""
    clean = re.sub(r"^https?://[^/]+/", "", path.strip()).strip("/").removesuffix(".git")
    if len(clean) > MAX_TEXT_SETTING:
        raise ValueError(path)
    return valid_project(clean)


def jira_key(project: str | None) -> str | None:
    key = text(project)
    if key is None:
        return None
    key = key.upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,9}", key):
        raise HTTPException(status_code=422, detail=f"{key!r} is not a Jira project key")
    return key


@router.get("/settings/connectors")
async def list_connectors(
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    config: Settings = Depends(get_settings),
) -> list[ConnectorStatus]:
    """GitHub, GitLab and Jira, each connected, not configured or failing, checked live."""
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


def text(value: str | None) -> str | None:
    """Trimmed; blank means unset."""
    if value is None or not (value := value.strip()):
        return None
    if len(value) > MAX_TEXT_SETTING:
        raise HTTPException(
            status_code=422, detail=f"Settings text must be at most {MAX_TEXT_SETTING} characters"
        )
    return value
