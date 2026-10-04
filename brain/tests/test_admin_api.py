"""Admin accounts: only an admin changes the connectors and creates accounts (POST /team/accounts).
Alex is the test team's admin; Sarah and Olga (on the other team) are not."""

import asyncio

import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM
from fastapi.testclient import TestClient
from test_auth import bearer

from brain.api.deps import current_user
from brain.auth import hash_password, verify_password
from brain.store import NotFound

ADMIN_ONLY = "Only an admin can do this"


def new_account(**changes) -> dict:
    return {"name": "Priya Natarajan", "email": "priya@example.com"} | changes


def run(coroutine):
    return asyncio.run(coroutine)


# who is an admin


def test_the_admin_flag_comes_with_me_and_the_members(client_as):
    assert client_as(ALEX).get("/me").json()["is_admin"] is True
    assert client_as(SARAH).get("/me").json()["is_admin"] is False

    members = {p["id"]: p["is_admin"] for p in client_as(SARAH).get("/team/members").json()}
    assert members == {ALEX.id: True, SARAH.id: False}


# creating accounts


def test_an_admin_creates_an_account_on_their_own_team(client_as, store):
    response = client_as(ALEX).post(
        "/team/accounts", json=new_account(name="  Priya   Natarajan ", title="Designer")
    )

    assert response.status_code == 201, response.text
    body = response.json()
    person = body["person"]
    assert (person["name"], person["short"], person["initials"]) == (
        "Priya Natarajan",
        "Priya",
        "PN",
    )
    assert (person["email"], person["title"], person["is_admin"]) == (
        "priya@example.com",
        "Designer",
        False,
    )
    assert person["id"] in run(store.team(TEAM.id)).member_ids
    assert person["id"] in {p["id"] for p in client_as(SARAH).get("/team/members").json()}
    assert person["id"] not in {p["id"] for p in client_as(OUTSIDER).get("/team/members").json()}
    login = run(store.login_by_email("priya@example.com"))
    assert login.person_id == person["id"]
    assert verify_password(login.password_hash, body["password"])


def test_the_generated_password_is_strong_shown_once_and_never_its_hash(client_as, store):
    alex = client_as(ALEX)

    first = alex.post("/team/accounts", json=new_account())
    second = alex.post("/team/accounts", json=new_account(email="sam@example.com"))

    passwords = [first.json()["password"], second.json()["password"]]
    assert all(len(p) >= 20 for p in passwords)
    assert passwords[0] != passwords[1]
    assert "argon2" not in first.text
    assert set(first.json()) == {"person", "password"}
    members = client_as(ALEX).get("/team/members").text
    assert passwords[0] not in members


def test_a_new_account_is_an_admin_only_when_asked(client_as):
    alex = client_as(ALEX)

    plain = alex.post("/team/accounts", json=new_account()).json()["person"]
    admin = alex.post(
        "/team/accounts", json=new_account(email="sam@example.com", name="Sam Lee", is_admin=True)
    ).json()["person"]

    assert (plain["is_admin"], plain["title"]) == (False, None)
    assert admin["is_admin"] is True


def test_the_new_account_signs_in_with_the_password_and_can_change_it(app, client_as):
    created = client_as(ALEX).post("/team/accounts", json=new_account()).json()
    del app.dependency_overrides[current_user]  # from here on, real sessions
    client = TestClient(app)

    login = client.post(
        "/auth/login", json={"email": "Priya@Example.com", "password": created["password"]}
    )

    assert login.status_code == 200, login.text
    headers = bearer(login.json()["token"])
    assert client.get("/me", headers=headers).json()["id"] == created["person"]["id"]
    changed = client.post(
        "/auth/password",
        json={"current_password": created["password"], "new_password": "priya-own-password"},
        headers=headers,
    )
    assert changed.status_code == 204, changed.text


def test_only_an_admin_creates_accounts(client_as, store):
    for person in (SARAH, OUTSIDER):
        response = client_as(person).post("/team/accounts", json=new_account())

        assert response.status_code == 403, person.id
        assert response.json()["detail"] == ADMIN_ONLY
    with pytest.raises(NotFound):
        run(store.login_by_email("priya@example.com"))
    assert len(run(store.team(TEAM.id)).member_ids) == 2


@pytest.mark.parametrize("email", ["sarah@example.com", "SARAH@Example.com ", "olga@example.com"])
def test_an_email_already_in_use_is_409(client_as, store, email):
    run(store.set_login(SARAH.id, "sarah@example.com", hash_password("sarah-password-1")))
    run(store.set_login(OUTSIDER.id, "olga@example.com", hash_password("olga-password-1")))

    response = client_as(ALEX).post("/team/accounts", json=new_account(email=email))

    assert response.status_code == 409, response.text
    assert len(run(store.team(TEAM.id)).member_ids) == 2
    assert verify_password(
        run(store.login_by_email(email.strip())).password_hash,
        "sarah-password-1" if "sarah" in email.lower() else "olga-password-1",
    )


def test_a_teammates_email_without_a_login_is_in_use_too(client_as, store):
    run(store.update_person(SARAH.model_copy(update={"email": "sarah@example.com"})))

    response = client_as(ALEX).post("/team/accounts", json=new_account(email="Sarah@example.com"))

    assert response.status_code == 409


@pytest.mark.parametrize(
    "changes",
    [
        {"email": "not-an-email"},
        {"email": ""},
        {"email": "two@@example.com"},
        {"email": "a b@example.com"},
        {"name": ""},
        {"name": "   "},
        {"name": "x" * 81},
    ],
)
def test_a_bad_email_or_name_is_422(client_as, store, changes):
    response = client_as(ALEX).post("/team/accounts", json=new_account(**changes))

    assert response.status_code == 422, changes
    assert isinstance(response.json()["detail"], str)
    assert len(run(store.team(TEAM.id)).member_ids) == 2


def test_a_malformed_body_is_422(client_as):
    assert client_as(ALEX).post("/team/accounts", json={"name": "No Email"}).status_code == 422


# admin status is read from the store on every request, not from the session


def test_revoking_an_admin_takes_effect_on_their_next_request(app, store):
    run(store.set_login(ALEX.id, "alex@example.com", hash_password("alex-password-1")))
    client = TestClient(app)
    login = client.post(
        "/auth/login", json={"email": "alex@example.com", "password": "alex-password-1"}
    )
    headers = bearer(login.json()["token"])
    assert login.json()["person"]["is_admin"] is True
    assert client.post("/team/accounts", json=new_account(), headers=headers).status_code == 201

    run(store.update_person(run(store.person(ALEX.id)).model_copy(update={"is_admin": False})))

    response = client.post(
        "/team/accounts", json=new_account(email="sam@example.com"), headers=headers
    )
    assert response.status_code == 403
    assert response.json()["detail"] == ADMIN_ONLY
    assert client.get("/me", headers=headers).json()["is_admin"] is False


def test_granting_admin_takes_effect_on_the_next_request(app, store):
    run(store.set_login(SARAH.id, "sarah@example.com", hash_password("sarah-password-1")))
    client = TestClient(app)
    token = client.post(
        "/auth/login", json={"email": "sarah@example.com", "password": "sarah-password-1"}
    ).json()["token"]
    assert (
        client.post("/team/accounts", json=new_account(), headers=bearer(token)).status_code == 403
    )

    run(store.update_person(run(store.person(SARAH.id)).model_copy(update={"is_admin": True})))

    assert (
        client.post("/team/accounts", json=new_account(), headers=bearer(token)).status_code == 201
    )


# team settings are the admin's; a person's own profile stays theirs


def team_settings(client: TestClient, **changes) -> dict:
    return client.get("/settings").json() | changes


@pytest.fixture
def linked(client_as):
    """The team has a repository and a Jira project, set by the admin."""
    alex = client_as(ALEX)
    body = team_settings(
        alex,
        github={"repo": "acme/checkout", "ref": "main"},
        jira={"site": "acme.atlassian.net", "project": "DS"},
        who_can_allow="host",
    )
    response = alex.put("/settings", json=body)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    "changes",
    [
        {"voice": "voice-2"},
        {"wake_phrase": "Hey Star"},
        {"sensitivity": "eager"},
        {"interrupt_minutes": 9},
        {"who_can_allow": "everyone"},
        {"timezone": "Europe/Berlin"},
        {"github": {"repo": "evil/fork", "ref": "main"}},
        {"jira": {"site": "acme.atlassian.net", "project": "EVIL"}},
        {},
    ],
)
def test_a_member_cannot_change_any_team_setting(client_as, linked, changes):
    sarah = client_as(SARAH)

    response = sarah.put("/settings", json=team_settings(sarah, **changes))

    assert response.status_code == 403
    assert response.json()["detail"] == ADMIN_ONLY
    assert sarah.get("/settings").json() == linked


def test_the_admin_changes_team_settings(client_as, linked):
    alex = client_as(ALEX)
    body = team_settings(alex, who_can_allow="everyone", voice="voice-2")
    body["github"] = body["github"] | {"repo": "acme/payments", "ref": None}

    response = alex.put("/settings", json=body)

    assert response.status_code == 200, response.text
    saved = client_as(SARAH).get("/settings").json()
    assert (saved["who_can_allow"], saved["voice"]) == ("everyone", "voice-2")
    assert (saved["github"]["repo"], saved["github"]["ref"]) == ("acme/payments", None)


def test_a_member_still_reads_the_team_settings_and_connectors(client_as, linked):
    sarah = client_as(SARAH)

    assert sarah.get("/settings").json() == linked
    assert sarah.get("/settings/connectors").status_code == 200


def test_a_member_still_changes_their_own_profile_and_photo(client_as):
    sarah = client_as(SARAH)

    assert sarah.patch("/me", json={"name": "Sarah K"}).json()["name"] == "Sarah K"
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    photo = sarah.post("/me/photo", files={"file": ("me.png", png, "image/png")})
    assert photo.status_code == 200, photo.text
    assert sarah.get("/me").json()["is_admin"] is False
