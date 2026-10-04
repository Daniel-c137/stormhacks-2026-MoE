"""An admin connects each of the team's GitHub repositories with the fine-grained personal access
token it is read with; that repository is then read through GitHub's hosted MCP server with its
token. A repository without one is read from GITHUB_MCP_URL with no credentials, and so are the
demo world's (MOCK_GITHUB_OWNERS), which take any token. Alex is the team's admin; Sarah is a
member. Nothing here reaches the network: GitHub's REST API is a stand-in transport and both MCP
servers are served on localhost."""

import asyncio

import anyio
import pytest
from api_support import ALEX, AUTH_SECRET, OTHER_TEAM, OUTSIDER, SARAH, TEAM
from fastapi.testclient import TestClient
from github_cloud_support import (
    EVERYTHING,
    LOGIN,
    OTHER_TOKEN,
    TOKEN,
    FakeGitHubApi,
    github_server,
    serve_with_token,
)

from brain.agent.ask import ToolOrchestrator
from brain.api.deps import get_http_transport, get_settings
from brain.sealing import Unsealable, unseal
from contracts import MAX_CODE_REPOS

ADMINS_ONLY = "Only an admin can do this"
REPO = "acme/checkout"
SITE = "acme/website"
DEMO = "dropsubs/dropsubs"  # the demo world's, served only by the mock


@pytest.fixture
def github(app) -> FakeGitHubApi:
    """The brain's outbound HTTP reaches only this GitHub REST API; the token reads both of
    Acme's repositories."""
    fake = FakeGitHubApi()
    fake.allow(REPO)
    fake.allow(SITE)
    app.dependency_overrides[get_http_transport] = lambda: fake.transport
    return fake


@pytest.fixture
def demo_world(settings):
    settings.mock_github_owners = "DropSubs"


@pytest.fixture
def hosted(settings):
    """GitHub's hosted MCP server: it answers only with the token."""
    with serve_with_token(github_server("hosted"), TOKEN) as (url, seen):
        settings.github_hosted_mcp_url = url
        yield seen


@pytest.fixture
def mock(settings):
    """The deployment's GITHUB_MCP_URL, the demo's mock: no credentials."""
    with serve_with_token(github_server("mock"), None) as (url, seen):
        settings.github_mcp_url = url
        yield seen


def connect(client: TestClient, repo: str = REPO, token: str = TOKEN, ref: str | None = None):
    return client.put("/settings/github/repos", json={"repo": repo, "ref": ref, "token": token})


def accounts(store) -> dict[str, object]:
    return {a.repo: a for a in asyncio.run(store.github_accounts(TEAM.id))}


def repos(response) -> list[tuple[str, str | None, str | None]]:
    return [(r["path"], r["ref"], r["login"]) for r in response.json()["github"]["repos"]]


def statuses(client: TestClient) -> dict[str, dict]:
    """GitHub's, by repository."""
    response = client.get("/settings/connectors")
    assert response.status_code == 200, response.text
    return {s["repo"]: s for s in response.json() if s["name"] == "github"}


# connecting


def test_an_admin_connects_a_repository_with_its_token(client_as, store, github):
    response = connect(client_as(ALEX))

    assert response.status_code == 200, response.text
    assert repos(response) == [(REPO, None, LOGIN)]
    saved = accounts(store)[REPO]
    assert (saved.login, saved.connected_by) == (LOGIN, ALEX.id)
    paths = [path.rstrip("/") for path in github.paths()]
    assert paths[0] == "/user"
    assert sorted(paths[1:]) == sorted(
        f"/repos/{REPO}{part}" for part in ("", "/issues", "/pulls", "/contents")
    )
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in github.requests)


def test_the_token_is_kept_sealed_and_never_sent_back(client_as, store, github):
    connected = connect(client_as(ALEX))
    seen_by_member = client_as(SARAH).get("/settings")
    status = client_as(SARAH).get("/settings/connectors")

    saved = accounts(store)[REPO]
    assert TOKEN not in saved.sealed_token
    assert unseal(saved.sealed_token, AUTH_SECRET, TEAM.id) == TOKEN
    with pytest.raises(Unsealable):  # copied into another team's row, it does not open
        unseal(saved.sealed_token, AUTH_SECRET, OTHER_TEAM.id)
    for response in (connected, seen_by_member, status):
        assert TOKEN not in response.text
    assert repos(seen_by_member) == [(REPO, None, LOGIN)]


def test_each_repository_is_checked_with_its_own_token_only(client_as, store, app):
    """Connecting one never depends on the others: their tokens are their own."""
    fake = FakeGitHubApi(tokens={TOKEN: LOGIN, OTHER_TOKEN: "site-bot"})
    fake.allow(REPO)
    app.dependency_overrides[get_http_transport] = lambda: fake.transport
    alex = client_as(ALEX)
    connect(alex)
    del fake.repos[REPO]  # the first token's repository is out of reach now
    fake.allow(SITE)
    fake.requests.clear()

    response = connect(alex, SITE, OTHER_TOKEN, ref="main")

    assert response.status_code == 200, response.text
    assert repos(response) == [(REPO, None, LOGIN), (SITE, "main", "site-bot")]
    assert not any(REPO in path for path in fake.paths())
    assert {a.login for a in accounts(store).values()} == {LOGIN, "site-bot"}


def test_a_pasted_link_is_kept_as_owner_and_name(client_as, store, github):
    response = connect(client_as(ALEX), f"https://github.com/{REPO}.git")

    assert response.status_code == 200, response.text
    assert repos(response) == [(REPO, None, LOGIN)]


def test_a_member_cannot_connect_a_repository(client_as, store, github):
    refused = connect(client_as(SARAH))

    assert refused.status_code == 403
    assert refused.json()["detail"] == ADMINS_ONLY
    assert accounts(store) == {}
    assert github.requests == []


def test_a_new_repository_needs_a_token(client_as, store, github):
    response = connect(client_as(ALEX), token="  ")

    assert response.status_code == 422
    assert "token" in response.json()["detail"]
    assert client_as(ALEX).get("/settings").json()["github"]["repos"] == []
    assert github.requests == []


def test_a_token_github_rejects_is_refused_and_nothing_is_saved(client_as, store, github):
    response = connect(client_as(ALEX), token=OTHER_TOKEN)

    assert response.status_code == 422
    assert "did not accept that token" in response.json()["detail"]
    assert OTHER_TOKEN not in response.text
    assert accounts(store) == {}
    assert client_as(ALEX).get("/settings").json()["github"]["repos"] == []


def test_a_token_that_cannot_see_the_repository_is_refused(client_as, store, github):
    response = connect(client_as(ALEX), "acme/secret")

    assert response.status_code == 422
    assert "acme/secret" in response.json()["detail"]
    assert accounts(store) == {}
    assert client_as(ALEX).get("/settings").json()["github"]["repos"] == []


@pytest.mark.parametrize(
    ("permission", "named"),
    [("issues", "Issues"), ("pull_requests", "Pull requests"), ("contents", "Contents")],
)
def test_a_token_missing_a_read_permission_names_the_repository_and_permission(
    client_as, store, github, permission, named
):
    github.allow(REPO, EVERYTHING - {permission})

    response = connect(client_as(ALEX))

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert REPO in detail and named in detail
    assert accounts(store) == {}


def test_an_empty_repository_is_readable(client_as, github):
    github.empty.add(REPO)

    assert connect(client_as(ALEX)).status_code == 200


def test_a_repository_with_issues_turned_off_is_readable(client_as, store, github):
    github.issues_off.add(REPO.casefold())

    response = connect(client_as(ALEX))

    assert response.status_code == 200, response.text
    assert accounts(store)[REPO].login == LOGIN


@pytest.mark.parametrize(
    "token",
    [
        "ghp_aClassicTokenWithBroadScopes00000000000",  # classic: not limited to repositories
        "github_pat_with spaces",
        "github_pat_" + "x" * 400,
    ],
)
def test_only_a_fine_grained_token_is_taken(client_as, store, github, token):
    response = connect(client_as(ALEX), token=token)

    assert response.status_code == 422
    assert "fine-grained" in response.json()["detail"]
    assert github.requests == []
    assert token not in response.text


@pytest.mark.parametrize("repo", ["acme", "acme/checkout/extra", "-acme/x", "a/.."])
def test_a_path_that_is_not_a_repository_is_refused(client_as, github, repo):
    response = connect(client_as(ALEX), repo)

    assert response.status_code == 422
    assert github.requests == []


def test_a_request_the_brain_cannot_read_does_not_echo_the_token(client_as, github):
    response = client_as(ALEX).put("/settings/github/repos", json={"repo": 12, "token": TOKEN})

    assert response.status_code == 422
    assert TOKEN not in response.text


def test_no_more_than_the_cap_of_repositories(client_as, store, github):
    alex = client_as(ALEX)
    full = [{"path": f"acme/repo-{i}"} for i in range(MAX_CODE_REPOS)]
    alex.put("/settings/connectors", json={"github": full})

    response = connect(alex)

    assert response.status_code == 422
    assert str(MAX_CODE_REPOS) in response.json()["detail"]
    assert github.requests == []


@pytest.mark.parametrize("trouble", ["down", "failing"])
def test_github_out_of_reach_is_a_502(client_as, store, github, trouble):
    setattr(github, trouble, True)

    response = connect(client_as(ALEX))

    assert response.status_code == 502
    assert accounts(store) == {}


@pytest.mark.parametrize("part", ["issues", "pull_requests", "contents"])
def test_github_failing_while_the_repository_is_checked_is_a_502(client_as, store, github, part):
    github.failing_parts = {part}

    response = connect(client_as(ALEX))

    assert response.status_code == 502, response.text
    assert accounts(store) == {}


def test_a_rate_limit_is_a_502_and_not_a_rejected_token(client_as, store, github):
    github.rate_limited = True

    response = connect(client_as(ALEX))

    assert response.status_code == 502, response.text
    assert "rate limit" in response.json()["detail"]
    assert accounts(store) == {}


# changing and removing


def test_a_new_branch_with_a_blank_token_keeps_the_token(client_as, store, github):
    alex = client_as(ALEX)
    connect(alex)
    before = accounts(store)[REPO]
    github.requests.clear()

    response = connect(alex, token="", ref="release")

    assert response.status_code == 200, response.text
    assert repos(response) == [(REPO, "release", LOGIN)]
    assert accounts(store)[REPO] == before
    assert github.requests == []


def test_a_new_token_replaces_the_repositorys_and_is_checked(client_as, store, app):
    fake = FakeGitHubApi(tokens={TOKEN: LOGIN, OTHER_TOKEN: "new-bot"})
    fake.allow(REPO)
    app.dependency_overrides[get_http_transport] = lambda: fake.transport
    alex = client_as(ALEX)
    connect(alex)

    response = connect(alex, token=OTHER_TOKEN)

    assert response.status_code == 200, response.text
    assert repos(response) == [(REPO, None, "new-bot")]
    assert unseal(accounts(store)[REPO].sealed_token, AUTH_SECRET, TEAM.id) == OTHER_TOKEN


def test_removing_a_repository_forgets_its_token(client_as, store, github):
    alex = client_as(ALEX)
    connect(alex)
    connect(alex, SITE)

    response = alex.put("/settings/connectors", json={"github": [{"path": SITE}]})

    assert response.status_code == 200, response.text
    assert repos(response) == [(SITE, None, LOGIN)]
    assert list(accounts(store)) == [SITE]


def test_saving_settings_or_connectors_keeps_the_tokens(client_as, store, github):
    alex = client_as(ALEX)
    connect(alex)
    current = alex.get("/settings").json()
    forged = {"repos": [{"path": REPO, "login": "someone-else"}]}

    saved = alex.put("/settings", json=current | {"github": forged})
    changed = alex.put("/settings/connectors", json={"github": [{"path": REPO, "ref": "dev"}]})

    assert repos(saved) == [(REPO, None, LOGIN)]
    assert repos(changed) == [(REPO, "dev", LOGIN)]
    assert accounts(store)[REPO].login == LOGIN


def test_a_repository_added_with_the_connectors_has_no_token(client_as, store, github):
    response = client_as(ALEX).put("/settings/connectors", json={"github": [{"path": REPO}]})

    assert response.status_code == 200, response.text
    assert repos(response) == [(REPO, None, None)]
    assert github.requests == []


def test_another_team_has_its_own_tokens(client_as, github):
    connect(client_as(ALEX))

    theirs = client_as(OUTSIDER).get("/settings").json()

    assert theirs["github"]["repos"] == []


# the demo world's repositories: any token, read from the mock


def test_a_demo_repository_takes_any_token_unchecked_and_unkept(
    client_as, store, github, demo_world
):
    response = connect(client_as(ALEX), DEMO, token="anything at all")

    assert response.status_code == 200, response.text
    assert repos(response) == [(DEMO, None, None)]
    assert github.requests == []
    assert accounts(store) == {}


def test_a_demo_repository_still_needs_something_in_the_token_field(client_as, demo_world):
    response = connect(client_as(ALEX), DEMO, token="")

    assert response.status_code == 422


def test_a_real_repository_is_never_checked_against_the_demo_ones(
    client_as, store, github, demo_world
):
    alex = client_as(ALEX)
    connect(alex, DEMO, token="x")

    response = connect(alex)

    assert response.status_code == 200, response.text
    assert repos(response) == [(DEMO, None, None), (REPO, None, LOGIN)]
    assert not any("dropsubs" in path for path in github.paths())


# status: one per repository


def test_each_repository_is_checked_where_it_is_read(
    client_as, store, github, hosted, mock, demo_world
):
    alex = client_as(ALEX)
    connect(alex)
    connect(alex, DEMO, token="anything")

    found = statuses(client_as(SARAH))

    assert list(found) == [REPO, DEMO]
    assert found[REPO]["state"] == "connected", found
    assert found[DEMO]["state"] == "connected", found
    assert TOKEN in hosted.tokens
    assert set(hosted.readonly) == {"true"}
    assert set(mock.tokens) == {None}


def test_a_token_github_no_longer_accepts_fails_only_its_repository(
    client_as, settings, github, mock
):
    alex = client_as(ALEX)
    connect(alex)
    alex.put("/settings/connectors", json={"github": [{"path": REPO}, {"path": SITE}]})
    with serve_with_token(github_server("hosted"), OTHER_TOKEN) as (url, _):  # token revoked
        settings.github_hosted_mcp_url = url

        found = statuses(alex)

    assert found[REPO]["state"] == "failing"
    assert "did not accept" in found[REPO]["detail"] and "401" in found[REPO]["detail"]
    assert TOKEN not in found[REPO]["detail"]
    assert found[SITE]["state"] == "connected"


def test_without_a_token_or_a_server_a_repository_is_not_set_up(client_as, settings):
    settings.github_mcp_url = None
    client_as(ALEX).put("/settings/connectors", json={"github": [{"path": REPO}]})

    found = statuses(client_as(ALEX))[REPO]

    assert found["state"] == "not_configured"
    assert "token" in found["detail"] and "GITHUB_MCP_URL" in found["detail"]


def test_without_a_repository_github_is_not_set_up(client_as, mock):
    response = client_as(ALEX).get("/settings/connectors")

    (github,) = [s for s in response.json() if s["name"] == "github"]
    assert github["state"] == "not_configured"
    assert github["repo"] is None
    assert "repository" in github["detail"]


def test_a_token_sealed_with_an_old_server_secret_is_not_reachable(
    app, client_as, settings, github, hosted
):
    connect(client_as(ALEX))
    changed = settings.model_copy(update={"auth_secret": AUTH_SECRET + "-rotated"})
    app.dependency_overrides[get_settings] = lambda: changed

    found = statuses(client_as(ALEX))[REPO]

    assert found["state"] == "failing"
    assert "connect it again" in found["detail"]
    assert hosted.tokens == []


# reading: a repository with a token is read through GitHub's server with it, others the mock


async def read_issue(store, settings, team_id: str, repo: str = REPO) -> tuple[list, list]:
    toolbox = await ToolOrchestrator(None, store, settings=settings).toolbox(team_id, ALEX.id)
    if isinstance(toolbox.github, str):
        return [], [toolbox.github]
    findings = await toolbox.github_read(number=41, kind="issue", repo=repo)
    return findings, []


def test_a_repository_with_a_token_is_read_through_githubs_server_with_it(
    client_as, store, settings, github, hosted, mock
):
    connect(client_as(ALEX))

    findings, problems = anyio.run(read_issue, store, settings, TEAM.id)

    assert problems == []
    assert [f.source.label for f in findings] == [f"{REPO}#41"]
    assert "hosted server" in findings[0].text
    assert hosted.tokens and set(hosted.tokens) == {TOKEN}
    assert mock.tokens == []


def test_a_repository_without_a_token_is_read_from_the_mock_beside_one_with(
    client_as, store, settings, github, hosted, mock
):
    alex = client_as(ALEX)
    connect(alex)
    alex.put("/settings/connectors", json={"github": [{"path": REPO}, {"path": SITE}]})

    findings, problems = anyio.run(read_issue, store, settings, TEAM.id, SITE)

    assert problems == []
    assert "mock server" in findings[0].text
    assert findings[0].source.label == f"{SITE}#41"
    assert set(mock.tokens) == {None}
    assert hosted.tokens == []


def test_a_demo_repository_is_read_from_the_mock(
    client_as, store, settings, github, hosted, mock, demo_world
):
    connect(client_as(ALEX), DEMO, token="anything")

    findings, problems = anyio.run(read_issue, store, settings, TEAM.id, DEMO)

    assert problems == []
    assert "mock server" in findings[0].text
    assert hosted.tokens == []


def test_after_removing_the_repository_its_token_is_not_used(
    client_as, store, settings, github, hosted, mock
):
    alex = client_as(ALEX)
    connect(alex)
    alex.put("/settings/connectors", json={"github": []})
    alex.put("/settings/connectors", json={"github": [{"path": REPO}]})  # back, without a token

    findings, _ = anyio.run(read_issue, store, settings, TEAM.id)

    assert "mock server" in findings[0].text
    assert hosted.tokens == []


def test_a_token_that_cannot_be_unsealed_never_falls_back_to_the_mock(
    client_as, store, settings, github, hosted, mock
):
    connect(client_as(ALEX))
    changed = settings.model_copy(update={"auth_secret": AUTH_SECRET + "-rotated"})

    findings, problems = anyio.run(read_issue, store, changed, TEAM.id)

    assert findings == []
    assert REPO in problems[0] and "connect it again" in problems[0]
    assert hosted.tokens == [] and mock.tokens == []
