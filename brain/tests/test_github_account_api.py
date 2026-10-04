"""An admin connects GitHub for the team with a fine-grained personal access token; the team's
reads then go to GitHub's hosted MCP server with that token. A team without one keeps reading
GITHUB_MCP_URL with no credentials (the demo's mock). Alex is the team's admin; Sarah is a
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

ADMINS_ONLY = "Only an admin can do this"
REPO = "acme/checkout"
SITE = "acme/website"


@pytest.fixture
def github(app) -> FakeGitHubApi:
    """The brain's outbound HTTP reaches only this GitHub REST API; the token reads both of the
    team's repositories."""
    fake = FakeGitHubApi()
    fake.allow(REPO)
    fake.allow(SITE)
    app.dependency_overrides[get_http_transport] = lambda: fake.transport
    return fake


@pytest.fixture
def repos(client_as):
    response = client_as(ALEX).put(
        "/settings/connectors", json={"github": [{"path": REPO}, {"path": SITE, "ref": "main"}]}
    )
    assert response.status_code == 200, response.text


@pytest.fixture
def hosted(settings):
    """GitHub's hosted MCP server: it answers only with the team's token."""
    with serve_with_token(github_server("hosted"), TOKEN) as (url, seen):
        settings.github_hosted_mcp_url = url
        yield seen


@pytest.fixture
def mock(settings):
    """The deployment's GITHUB_MCP_URL, the demo's mock: no credentials."""
    with serve_with_token(github_server("mock"), None) as (url, seen):
        settings.github_mcp_url = url
        yield seen


def connect(client: TestClient, token: str = TOKEN):
    return client.put("/settings/github/account", json={"token": token})


def account(store):
    return asyncio.run(store.github_account(TEAM.id))


def statuses(client: TestClient) -> dict[str, dict]:
    response = client.get("/settings/connectors")
    assert response.status_code == 200, response.text
    return {s["name"]: s for s in response.json()}


# connecting


def test_an_admin_connects_github_with_a_token(client_as, store, github, repos):
    response = connect(client_as(ALEX))

    assert response.status_code == 200, response.text
    assert response.json()["github"]["account_login"] == LOGIN
    saved = account(store)
    assert (saved.login, saved.connected_by) == (LOGIN, ALEX.id)
    paths = [path.rstrip("/") for path in github.paths()]
    assert paths[0] == "/user"
    for repo in (REPO, SITE):
        for part in ("", "/issues", "/pulls", "/contents"):
            assert f"/repos/{repo}{part}" in paths
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in github.requests)


def test_the_token_is_kept_sealed_and_never_sent_back(client_as, store, github, repos):
    connected = connect(client_as(ALEX))
    seen_by_member = client_as(SARAH).get("/settings")
    status = client_as(SARAH).get("/settings/connectors")

    saved = account(store)
    assert TOKEN not in saved.sealed_token
    assert unseal(saved.sealed_token, AUTH_SECRET, TEAM.id) == TOKEN
    with pytest.raises(Unsealable):  # copied into another team's row, it does not open
        unseal(saved.sealed_token, AUTH_SECRET, OTHER_TEAM.id)
    for response in (connected, seen_by_member, status):
        assert TOKEN not in response.text
    assert seen_by_member.json()["github"]["account_login"] == LOGIN


def test_a_member_cannot_connect_or_disconnect(client_as, store, github, repos):
    sarah = client_as(SARAH)

    refused = connect(sarah)
    connect(client_as(ALEX))
    kept = sarah.delete("/settings/github/account")

    assert (refused.status_code, kept.status_code) == (403, 403)
    assert refused.json()["detail"] == ADMINS_ONLY
    assert account(store) is not None
    assert github.paths().count("/user") == 1  # the member's attempt never reached GitHub


def test_a_token_github_rejects_is_refused_and_nothing_is_saved(client_as, store, github, repos):
    response = connect(client_as(ALEX), OTHER_TOKEN)

    assert response.status_code == 422
    assert "did not accept that token" in response.json()["detail"]
    assert OTHER_TOKEN not in response.text
    assert account(store) is None
    assert client_as(ALEX).get("/settings").json()["github"]["account_login"] is None


def test_a_token_that_cannot_see_a_connected_repository_is_refused(client_as, store, github, repos):
    del github.repos[SITE]

    response = connect(client_as(ALEX))

    assert response.status_code == 422
    assert SITE in response.json()["detail"]
    assert account(store) is None


@pytest.mark.parametrize(
    ("permission", "named"),
    [("issues", "Issues"), ("pull_requests", "Pull requests"), ("contents", "Contents")],
)
def test_a_token_missing_a_read_permission_names_the_repository_and_permission(
    client_as, store, github, repos, permission, named
):
    github.allow(SITE, EVERYTHING - {permission})

    response = connect(client_as(ALEX))

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert SITE in detail and named in detail
    assert account(store) is None


def test_an_empty_repository_is_readable(client_as, store, github, repos):
    github.empty.add(SITE)

    response = connect(client_as(ALEX))

    assert response.status_code == 200, response.text


def test_a_team_without_repositories_connects_on_the_token_alone(client_as, store, github):
    response = connect(client_as(ALEX))

    assert response.status_code == 200, response.text
    assert github.paths() == ["/user"]


@pytest.mark.parametrize(
    "token",
    [
        "ghp_aClassicTokenWithBroadScopes00000000000",  # classic: not limited to repositories
        "",
        "github_pat_with spaces",
        "github_pat_" + "x" * 400,
    ],
)
def test_only_a_fine_grained_token_is_taken(client_as, store, github, token):
    response = connect(client_as(ALEX), token)

    assert response.status_code == 422
    assert "fine-grained" in response.json()["detail"]
    assert github.requests == []
    if token:
        assert token not in response.text


def test_a_request_the_brain_cannot_read_does_not_echo_the_token(client_as, github):
    response = client_as(ALEX).put("/settings/github/account", json={"token": 12, "x": TOKEN})

    assert response.status_code == 422
    assert TOKEN not in response.text


@pytest.mark.parametrize("trouble", ["down", "failing"])
def test_github_out_of_reach_is_a_502(client_as, store, github, repos, trouble):
    setattr(github, trouble, True)

    response = connect(client_as(ALEX))

    assert response.status_code == 502
    assert account(store) is None


def test_an_admin_disconnects_github(client_as, store, github, repos):
    alex = client_as(ALEX)
    connect(alex)

    response = alex.delete("/settings/github/account")

    assert response.status_code == 200, response.text
    assert response.json()["github"]["account_login"] is None
    assert [r["path"] for r in response.json()["github"]["repos"]] == [REPO, SITE]
    assert account(store) is None


def test_saving_settings_or_connectors_keeps_the_connection(client_as, store, github, repos):
    alex = client_as(ALEX)
    connect(alex)
    current = alex.get("/settings").json()

    saved = alex.put("/settings", json=current | {"github": {"repos": [], "account_login": "x"}})
    changed = alex.put("/settings/connectors", json={"github": [{"path": REPO}]})

    assert saved.json()["github"]["account_login"] == LOGIN
    assert changed.json()["github"]["account_login"] == LOGIN
    assert account(store).login == LOGIN


def test_a_repository_added_later_must_be_readable_with_the_token(client_as, store, github, repos):
    alex = client_as(ALEX)
    connect(alex)

    refused = alex.put(
        "/settings/connectors", json={"github": [{"path": REPO}, {"path": "acme/secret"}]}
    )
    github.allow("acme/secret")
    accepted = alex.put(
        "/settings/connectors", json={"github": [{"path": REPO}, {"path": "acme/secret"}]}
    )

    assert refused.status_code == 422
    assert "acme/secret" in refused.json()["detail"]
    assert accepted.status_code == 200, accepted.text


def test_another_team_has_its_own_connection(client_as, github):
    connect(client_as(ALEX))

    theirs = client_as(OUTSIDER).get("/settings").json()

    assert theirs["github"]["account_login"] is None


# status


def test_with_a_token_the_status_is_checked_on_githubs_server_with_it(
    client_as, github, repos, hosted, mock
):
    connect(client_as(ALEX))

    found = statuses(client_as(SARAH))["github"]

    assert found["state"] == "connected", found
    assert TOKEN in hosted.tokens
    assert set(hosted.readonly) == {"true"}
    assert mock.tokens == []  # the mock is not asked once a token is connected


def test_a_token_github_no_longer_accepts_is_not_reachable_with_a_reason(
    client_as, settings, github, repos, mock
):
    connect(client_as(ALEX))
    with serve_with_token(github_server("hosted"), OTHER_TOKEN) as (url, _):  # token revoked
        settings.github_hosted_mcp_url = url

        found = statuses(client_as(ALEX))["github"]

    assert found["state"] == "failing"
    assert "did not accept" in found["detail"] and "401" in found["detail"]
    assert TOKEN not in found["detail"]


def test_without_a_token_the_status_is_the_mocks_without_credentials(client_as, repos, mock):
    found = statuses(client_as(ALEX))["github"]

    assert found["state"] == "connected", found
    assert mock.tokens and set(mock.tokens) == {None}


def test_without_a_token_or_a_server_github_is_not_set_up(client_as, settings, repos):
    settings.github_mcp_url = None

    found = statuses(client_as(ALEX))["github"]

    assert found["state"] == "not_configured"
    assert "Connect GitHub" in found["detail"]


def test_with_a_token_but_no_repository_github_is_not_set_up(client_as, github, hosted):
    connect(client_as(ALEX))

    found = statuses(client_as(ALEX))["github"]

    assert found["state"] == "not_configured"
    assert "repository" in found["detail"]


def test_a_token_sealed_with_an_old_server_secret_is_not_reachable(
    app, client_as, settings, github, repos, hosted
):
    connect(client_as(ALEX))
    changed = settings.model_copy(update={"auth_secret": AUTH_SECRET + "-rotated"})
    app.dependency_overrides[get_settings] = lambda: changed

    found = statuses(client_as(ALEX))["github"]

    assert found["state"] == "failing"
    assert "connect GitHub again" in found["detail"]
    assert hosted.tokens == []


# reading: the team with a token reads GitHub's server with it; one without reads the mock


async def read_issue(store, settings, team_id: str) -> tuple[list, list]:
    toolbox = await ToolOrchestrator(None, store, settings=settings).toolbox(team_id, ALEX.id)
    if isinstance(toolbox.github, str):
        return [], [toolbox.github]
    findings = await toolbox.github_read(number=41, kind="issue", repo=REPO)
    return findings, []


def test_a_team_with_a_token_reads_through_githubs_server_with_it(
    client_as, store, settings, github, repos, hosted, mock
):
    connect(client_as(ALEX))

    findings, problems = anyio.run(read_issue, store, settings, TEAM.id)

    assert problems == []
    assert [f.source.label for f in findings] == [f"{REPO}#41"]
    assert "hosted server" in findings[0].text
    assert hosted.tokens and set(hosted.tokens) == {TOKEN}
    assert mock.tokens == []


def test_a_team_without_a_token_reads_the_mock_without_credentials(
    client_as, store, settings, github, repos, hosted, mock
):
    findings, problems = anyio.run(read_issue, store, settings, TEAM.id)

    assert problems == []
    assert "mock server" in findings[0].text
    assert findings[0].source.label == f"{REPO}#41"
    assert hosted.tokens == []
    assert set(mock.tokens) == {None}


def test_after_disconnecting_the_team_reads_the_mock_again(
    client_as, store, settings, github, repos, hosted, mock
):
    alex = client_as(ALEX)
    connect(alex)
    alex.delete("/settings/github/account")

    findings, _ = anyio.run(read_issue, store, settings, TEAM.id)

    assert "mock server" in findings[0].text
    assert hosted.tokens == []


def test_a_token_that_cannot_be_unsealed_never_falls_back_to_the_mock(
    client_as, store, settings, github, repos, hosted, mock
):
    connect(client_as(ALEX))
    changed = settings.model_copy(update={"auth_secret": AUTH_SECRET + "-rotated"})

    findings, problems = anyio.run(read_issue, store, changed, TEAM.id)

    assert findings == []
    assert "connect GitHub again" in problems[0]
    assert hosted.tokens == [] and mock.tokens == []
