"""PUT /settings/connectors: an admin connects any number of GitHub repositories and GitLab
projects (up to MAX_CODE_REPOS each) and the Jira site and project. Runs on both stores."""

import asyncio
from datetime import UTC, datetime

import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM
from fastapi import HTTPException

from brain.api.deps import require_admin
from contracts import MAX_CODE_REPOS, CodeRepo, GitHubSettings, TeamSettings

ADMIN_ONLY = "Only an admin can do this"


def body(**changes) -> dict:
    return {
        "github": [{"path": "acme/checkout", "ref": "main"}, {"path": "acme/website"}],
        "gitlab": [{"path": "acme/platform/infra"}],
        "jira": {"site": "acme.atlassian.net", "project": "DS"},
    } | changes


def put(client, **changes):
    return client.put("/settings/connectors", json=body(**changes))


def test_an_admin_connects_several_repositories_a_gitlab_project_and_jira(client_as):
    response = put(client_as(ALEX))

    assert response.status_code == 200, response.text
    saved = client_as(SARAH).get("/settings").json()
    assert saved == response.json()
    assert [(r["path"], r["ref"]) for r in saved["github"]["repos"]] == [
        ("acme/checkout", "main"),
        ("acme/website", None),
    ]
    assert [p["path"] for p in saved["gitlab"]["projects"]] == ["acme/platform/infra"]
    assert (saved["jira"]["site"], saved["jira"]["project"]) == ("acme.atlassian.net", "DS")
    assert not any(r["connected"] for r in saved["github"]["repos"])


def test_removing_a_repository_and_adding_none_is_a_plain_replace(client_as):
    admin = client_as(ALEX)
    put(admin)

    saved = put(admin, github=[{"path": "acme/website"}], gitlab=[]).json()

    assert [r["path"] for r in saved["github"]["repos"]] == ["acme/website"]
    assert saved["gitlab"]["projects"] == []


def test_the_other_settings_are_kept(client_as):
    alex = client_as(ALEX)
    settings = alex.get("/settings").json() | {"voice": "voice-1", "timezone": "Europe/Berlin"}
    assert alex.put("/settings", json=settings).status_code == 200

    saved = put(client_as(ALEX)).json()

    assert (saved["voice"], saved["timezone"]) == ("voice-1", "Europe/Berlin")


def test_only_an_admin_changes_the_connectors(client_as):
    response = put(client_as(SARAH))

    assert response.status_code == 403
    assert response.json()["detail"] == ADMIN_ONLY
    assert client_as(ALEX).get("/settings").json()["github"]["repos"] == []


def test_require_admin_refuses_anyone_without_the_flag():
    assert asyncio.run(require_admin(ALEX)) is ALEX  # Alex is the test team's admin
    with pytest.raises(HTTPException) as refused:
        asyncio.run(require_admin(SARAH))
    assert (refused.value.status_code, refused.value.detail) == (403, ADMIN_ONLY)


def test_an_admin_changes_only_their_own_team(client_as):
    other_admin = OUTSIDER.model_copy(update={"is_admin": True})

    assert put(client_as(other_admin)).status_code == 200

    assert client_as(ALEX).get("/settings").json()["github"]["repos"] == []


def test_connection_and_index_state_stay_with_a_repository_that_stays(client_as, store):
    indexed = datetime(2026, 10, 1, 12, tzinfo=UTC)
    asyncio.run(
        store.save_settings(
            TeamSettings(
                team_id=TEAM.id,
                github=GitHubSettings(
                    repos=[
                        CodeRepo(
                            path="acme/checkout",
                            ref="main",
                            connected=True,
                            files=420,
                            indexed_at=indexed,
                        ),
                        CodeRepo(path="acme/website", connected=True, files=12),
                    ]
                ),
            )
        )
    )

    saved = put(
        client_as(ALEX),
        github=[
            {"path": "ACME/checkout", "ref": "main", "connected": False},  # client state ignored
            {"path": "acme/website", "ref": "launch"},  # a new ref is a new index
            {"path": "acme/new"},
        ],
    ).json()

    checkout, website, new = saved["github"]["repos"]
    assert (checkout["path"], checkout["connected"], checkout["files"]) == (
        "ACME/checkout",
        True,
        420,
    )
    assert checkout["indexed_at"].startswith("2026-10-01T12:00:00")
    assert (website["ref"], website["connected"], website["files"]) == ("launch", False, None)
    assert (new["connected"], new["files"]) == (False, None)


@pytest.mark.parametrize(
    ("given", "kept"),
    [
        ("https://github.com/acme/web", "acme/web"),
        ("github.com/acme/web.git", "acme/web"),
        ("  acme/web/ ", "acme/web"),
    ],
)
def test_github_links_are_kept_as_owner_and_name(client_as, given, kept):
    saved = put(client_as(ALEX), github=[{"path": given}]).json()

    assert [r["path"] for r in saved["github"]["repos"]] == [kept]


def test_gitlab_links_are_kept_as_the_project_path(client_as):
    link = "https://gitlab.example.com/acme/platform/infra.git"

    saved = put(client_as(ALEX), gitlab=[{"path": link}]).json()

    assert [p["path"] for p in saved["gitlab"]["projects"]] == ["acme/platform/infra"]


@pytest.mark.parametrize(
    ("site", "kept"),
    [
        ("https://acme.atlassian.net", "acme.atlassian.net"),
        ("https://acme.atlassian.net/", "acme.atlassian.net"),
        ("HTTP://jira.acme.example/jira/", "jira.acme.example/jira"),
        ("acme.atlassian.net", "acme.atlassian.net"),
        ("  ", None),
    ],
)
def test_the_jira_site_is_kept_without_its_scheme_so_links_work(client_as, site, kept):
    saved = put(client_as(ALEX), jira={"site": site, "project": "ds"}).json()

    assert saved["jira"]["site"] == kept
    assert saved["jira"]["project"] == "DS"


@pytest.mark.parametrize(
    "changes",
    [
        {"github": [{"path": "just-a-name"}]},
        {"github": [{"path": "owner/name/extra"}]},
        {"github": [{"path": "owner/.."}]},
        {"github": [{"path": "acme/web"}, {"path": "ACME/web"}]},
        {"gitlab": [{"path": "infra"}]},
        {"gitlab": [{"path": "acme/../infra"}]},
        {"github": [{"path": f"acme/r{n}"} for n in range(MAX_CODE_REPOS + 1)]},
        {"gitlab": [{"path": f"acme/p{n}"} for n in range(MAX_CODE_REPOS + 1)]},
        {"jira": {"project": "not a key"}},
        {"github": [{"path": "acme/web", "ref": "x" * 201}]},
    ],
)
def test_bad_connectors_are_refused_and_nothing_is_saved(client_as, changes):
    response = put(client_as(ALEX), **changes)

    assert response.status_code == 422, response.text
    assert client_as(ALEX).get("/settings").json()["github"]["repos"] == []


def test_the_cap_itself_is_accepted(client_as):
    repos = [{"path": f"acme/r{n}"} for n in range(MAX_CODE_REPOS)]

    saved = put(client_as(ALEX), github=repos, gitlab=[]).json()

    assert len(saved["github"]["repos"]) == MAX_CODE_REPOS
