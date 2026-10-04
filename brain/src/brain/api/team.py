"""Profile, team, members and workspace settings (board -> brain)."""

import re
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile

from contracts import (
    MAX_CODE_REPOS,
    CodeRepo,
    CodeRepoChoice,
    ConnectorStatus,
    ConnectorsUpdate,
    GitHubRepoConnect,
    JiraAccountConnect,
    Person,
    ProfileUpdate,
    Team,
    TeamSettings,
    Voice,
)

from ..accounts import signs_in
from ..auth import NOT_CONFIGURED, signing_secret
from ..config import Settings
from ..connectors import connector_statuses
from ..github_account import (
    NOT_FINE_GRAINED,
    GitHubApi,
    GitHubRejected,
    GitHubUnreachable,
    fine_grained,
    github_endpoints,
)
from ..gitlab import valid_project
from ..jira import site_host
from ..jira_rest import (
    NOT_A_SITE,
    JiraAccess,
    JiraCloud,
    JiraRejected,
    JiraUnreachable,
    account_access,
    issue_type_for_tasks,
    reads_with_account,
)
from ..sealing import seal
from ..store import GitHubAccount, JiraAccount, NotFound, Store
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


NO_JIRA_ACCOUNT = {
    "connected": False,
    "account_email": None,
    "account_site": None,
    "account_project": None,
}


async def shown(store: Store, settings: TeamSettings) -> TeamSettings:
    """Settings as they are sent: what they say about the Jira and GitHub accounts is read from
    the saved accounts themselves, the ones pushes and reads use, so the two can never
    disagree. Each GitHub repository names the account its token is from."""
    account = await store.jira_account(settings.team_id)
    about = NO_JIRA_ACCOUNT
    if account is not None:
        about = {
            "connected": True,
            "account_email": account.email,
            "account_site": account.site,
            "account_project": account.project,
        }
    logins = {a.repo: a.login for a in await store.github_accounts(settings.team_id)}
    repos = [
        repo.model_copy(update={"login": logins.get(repo.path.lower())})
        for repo in settings.github.repos
    ]
    return settings.model_copy(
        update={
            "jira": settings.jira.model_copy(update=about),
            "github": settings.github.model_copy(update={"repos": repos}),
        }
    )


@router.get("/settings")
async def read_team_settings(
    user: Person = Depends(current_user), store: Store = Depends(get_store)
) -> TeamSettings:
    team = await user_team(store, user)
    return await shown(store, await store.settings(team.id))


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
    saved = await store.save_settings(
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
    return await shown(store, saved)


@router.put("/settings/connectors")
async def write_connectors(
    body: ConnectorsUpdate,
    user: Person = Depends(require_admin),
    store: Store = Depends(get_store),
) -> TeamSettings:
    """The team's GitHub repositories, GitLab projects and Jira site and project, replacing
    the saved ones; admins only. A repository that stays keeps its connection and index state
    (a new branch or tag drops its index) and its token; a repository removed loses its token.
    One added here has none: it is read from GITHUB_MCP_URL with no credentials (PUT
    /settings/github/repos connects one with its token). Paths are checked and repeats refused.
    The Jira site and project here are what the agent reads; the account connected for the
    team's project (PUT /settings/jira/account) stays as it is."""
    team = await user_team(store, user)
    current = await store.settings(team.id)
    github = repos(body.github, current.github.repos, "GitHub repository", github_path)
    gitlab = repos(body.gitlab, current.gitlab.projects, "GitLab project", gitlab_path)
    jira = current.jira.model_copy(
        update={"site": site_host(text(body.jira.site)), "project": jira_key(body.jira.project)}
        | NO_JIRA_ACCOUNT  # the account is not kept here: `shown` reads it from where it is
    )
    saved = await store.save_settings(
        current.model_copy(
            update={
                "github": current.github.model_copy(update={"repos": without_logins(github)}),
                "gitlab": current.gitlab.model_copy(update={"projects": gitlab}),
                "jira": jira,
            }
        )
    )
    kept = {repo.path.lower() for repo in github}
    for account in await store.github_accounts(team.id):
        if account.repo not in kept:
            await store.delete_github_account(team.id, account.repo)
    return await shown(store, saved)


def without_logins(github: list[CodeRepo]) -> list[CodeRepo]:
    """As saved: whose token a repository is read with is read from its account (`shown`)."""
    return [repo.model_copy(update={"login": None}) for repo in github]


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


MAX_API_TOKEN = 2000


@router.put("/settings/jira/account")
async def connect_jira_account(
    body: JiraAccountConnect,
    user: Person = Depends(require_admin),
    store: Store = Depends(get_store),
    config: Settings = Depends(get_settings),
    transport: httpx.AsyncBaseTransport | None = Depends(get_http_transport),
) -> TeamSettings:
    """Connects the team's Jira project with the account it is read and pushed with; admins
    only. Before anything is saved, Jira is asked whether it accepts the email and API token,
    whether that account sees the project and may create issues in it, and which issue type
    tasks become there. The token is stored encrypted and never returned. The project becomes
    the one the agent reads, on its site with that account, and approved task drafts are
    created there as issues as that account.

    One of the demo world's projects (MOCK_JIRA_PROJECTS) takes any account: nothing is
    checked or kept, it becomes the project the agent reads, and it is read and pushed through
    JIRA_MCP_URL, the mock, as before."""
    email, token = body.email.strip(), body.api_token.strip()
    project = jira_key(body.project)
    if not email or not token or project is None:
        raise HTTPException(
            status_code=422, detail="The account's email, its API token and a project are needed"
        )
    if len(email) > MAX_TEXT_SETTING or len(token) > MAX_API_TOKEN:
        raise HTTPException(status_code=422, detail="That email or API token is too long")
    try:
        access = JiraAccess(site=body.site, email=email, api_token=token, project_key=project)
    except ValueError:
        raise HTTPException(status_code=422, detail=NOT_A_SITE) from None
    team = await user_team(store, user)
    if config.mocks_jira_project(project):
        await store.delete_jira_account(team.id)
        return await shown(store, await read_jira_project(store, team.id, access.site, project))
    if (secret := signing_secret(config)) is None:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)
    issue_type = await checked_with_jira(JiraCloud(access, transport=transport))

    await store.save_jira_account(
        JiraAccount(
            team_id=team.id,
            site=access.site,
            project=project,
            issue_type_id=issue_type,
            email=email,
            sealed_token=seal(token, secret, team.id),
            connected_by=user.id,
            connected_at=datetime.now(UTC),
        )
    )
    return await shown(store, await read_jira_project(store, team.id, access.site, project))


async def read_jira_project(store: Store, team_id: str, site: str, project: str) -> TeamSettings:
    """Makes the project the one the agent reads."""
    current = await store.settings(team_id)
    jira = current.jira.model_copy(update={"site": site, "project": project} | NO_JIRA_ACCOUNT)
    return await store.save_settings(current.model_copy(update={"jira": jira}))


async def checked_with_jira(cloud: JiraCloud) -> str | None:
    """The id of the issue type tasks are created as in the project (None when Jira does not
    list its types: the type named Task is then used). 422 when the account or the project
    cannot be used as given, 502 when Jira cannot be asked."""
    site, project = cloud.access.site, cloud.access.project_key

    def refuse(detail: str) -> HTTPException:
        return HTTPException(status_code=422, detail=detail)

    try:
        try:
            await cloud.myself()
        except JiraRejected as e:
            if e.status == 404:
                raise refuse(f"There is no Jira site at {site}") from None
            if e.status in (401, 403):
                raise refuse(f"{site} did not accept that email and API token") from None
            raise
        try:
            found = await cloud.project(project)
        except JiraRejected as e:
            if e.status in (403, 404):
                raise refuse(f"{site} has no project {project} that this account can see") from None
            raise
        issue_type = issue_type_for_tasks(found)
        if issue_type is None and isinstance(found.get("issueTypes"), list):
            raise refuse(f"{project} has no issue type that tasks can be created as")
        try:
            allowed = await cloud.can_create(project)
        except JiraRejected:
            allowed = True  # Jira would not say; a push reports its own refusal
        if not allowed:
            raise refuse(f"This account is not allowed to create issues in {project}")
    except JiraUnreachable as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    except JiraRejected as e:
        raise HTTPException(status_code=502, detail=f"{site} answered {e.status}: {e}") from None
    return issue_type


@router.delete("/settings/jira/account")
async def disconnect_jira_account(
    user: Person = Depends(require_admin), store: Store = Depends(get_store)
) -> TeamSettings:
    """Removes the team's Jira project; admins only: its account and token are forgotten, and
    the agent stops reading the project when it is the account's. A project the agent reads
    apart from the account (connected before the two were one) stays."""
    team = await user_team(store, user)
    account = await store.jira_account(team.id)
    await store.delete_jira_account(team.id)
    current = await store.settings(team.id)
    if account is not None and same_project(current, account):
        jira = current.jira.model_copy(update={"site": None, "project": None})
        current = await store.save_settings(current.model_copy(update={"jira": jira}))
    return await shown(store, current)


def same_project(team: TeamSettings, account: JiraAccount) -> bool:
    """Whether the project the agent reads is the one the account was connected with."""
    project, site = team.jira.project, team.jira.site
    return (
        project is not None
        and project.upper() == account.project.upper()
        and (site is None or site.casefold() == account.site.casefold())
    )


@router.put("/settings/github/repos")
async def connect_github_repo(
    body: GitHubRepoConnect,
    user: Person = Depends(require_admin),
    store: Store = Depends(get_store),
    config: Settings = Depends(get_settings),
    transport: httpx.AsyncBaseTransport | None = Depends(get_http_transport),
) -> TeamSettings:
    """Connects one of the team's GitHub repositories with the fine-grained personal access
    token it is read with, or changes the branch or tag or the token of one already connected
    (the same path); admins only. Before anything is saved, GitHub is asked whose token it is
    and whether it reads the repository (its metadata, issues, pull requests and code). The
    token is stored encrypted and never returned; the repository is then read through GitHub's
    hosted MCP server with it. Each repository has its own token, so connecting one never
    depends on the others. A blank token keeps the one a connected repository has.

    The demo world's repositories (MOCK_GITHUB_OWNERS) take any token: it is not checked or
    kept, and they are read from GITHUB_MCP_URL, the mock, as before."""
    try:
        path = github_path(body.repo)
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"{body.repo.strip()!r} is not a GitHub repository path"
        ) from None
    token = body.token.strip()
    team = await user_team(store, user)
    current = await store.settings(team.id)
    chosen = [CodeRepoChoice(path=r.path, ref=r.ref) for r in current.github.repos]
    known = [i for i, r in enumerate(chosen) if r.path.casefold() == path.casefold()]
    if known:
        chosen[known[0]] = CodeRepoChoice(path=path, ref=body.ref)
    elif not token:
        raise HTTPException(status_code=422, detail="Paste the token the repository is read with")
    elif len(chosen) >= MAX_CODE_REPOS:
        raise HTTPException(
            status_code=422, detail=f"At most {MAX_CODE_REPOS} repositories can be connected"
        )
    else:
        chosen.append(CodeRepoChoice(path=path, ref=body.ref))
    github = repos(chosen, current.github.repos, "GitHub repository", github_path)
    mock = config.mocks_repository(path)
    account = None
    if token and not mock:
        account = await github_account(config, transport, team.id, path, token, user.id)

    saved = await store.save_settings(
        current.model_copy(
            update={"github": current.github.model_copy(update={"repos": without_logins(github)})}
        )
    )
    if account is not None:
        await store.save_github_account(account)
    elif mock:  # read from the mock whatever was kept for it before
        await store.delete_github_account(team.id, path.lower())
    return await shown(store, saved)


async def github_account(
    config: Settings,
    transport: httpx.AsyncBaseTransport | None,
    team_id: str,
    path: str,
    token: str,
    by: str,
) -> GitHubAccount:
    """The repository's account, its token sealed, once GitHub said whose token it is and that
    it reads the repository. 422 when it is not a fine-grained token, GitHub refuses it or it
    can't read the repository (naming the permission it lacks); 502 when GitHub can't be
    asked; 503 without AUTH_SECRET."""
    if not fine_grained(token):
        raise HTTPException(status_code=422, detail=NOT_FINE_GRAINED)
    if (secret := signing_secret(config)) is None:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)
    api = GitHubApi(config.github_api_url, token, transport)
    try:
        login = await api.login()
        problem = await api.unreadable(path)
    except GitHubRejected as e:
        if e.status in (401, 403):
            raise HTTPException(
                status_code=422, detail=f"GitHub did not accept that token ({e.status}: {e})"
            ) from None
        raise HTTPException(status_code=502, detail=f"GitHub answered {e.status}") from None
    except GitHubUnreachable as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    return GitHubAccount(
        team_id=team_id,
        repo=path.lower(),
        login=login,
        sealed_token=seal(token, secret, team_id),
        connected_by=by,
        connected_at=datetime.now(UTC),
    )


@router.get("/settings/connectors")
async def list_connectors(
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    config: Settings = Depends(get_settings),
    transport: httpx.AsyncBaseTransport | None = Depends(get_http_transport),
) -> list[ConnectorStatus]:
    """Each GitHub repository, GitLab and Jira, each connected, not configured or failing,
    checked live. A repository connected with a token is checked on GitHub's hosted server with
    it; a Jira project connected with an account, on its own site as that account."""
    team = await user_team(store, user)
    settings = await store.settings(team.id)
    github = github_endpoints(config, await store.github_accounts(team.id))
    jira = jira_site(config, settings, await store.jira_account(team.id), transport)
    return await connector_statuses(config, settings, github, jira)


JIRA_CONNECT_AGAIN = (
    "The token the project was connected with can't be read on this server: an admin must "
    "connect Jira again"
)


def jira_site(
    config: Settings,
    team: TeamSettings,
    account: JiraAccount | None,
    transport: httpx.AsyncBaseTransport | None,
) -> JiraCloud | str | None:
    """The team's Jira site as its account, when the agent reads the project there; why the
    account can't be used; None when the project is read through JIRA_MCP_URL."""
    if account is None or not reads_with_account(
        config, team.jira.project, team.jira.site, account
    ):
        return None
    access = account_access(config, account)
    return JIRA_CONNECT_AGAIN if access is None else JiraCloud(access, transport=transport)


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
