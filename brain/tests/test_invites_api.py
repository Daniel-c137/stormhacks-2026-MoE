"""Inviting someone (#143): an admin adds a person to their own team with an email and no login
(POST /team/accounts with invite). That person then creates their account on the sign-in page,
with a password or Google, and joins the team that invited them. Until then the members list
marks them invited."""

import asyncio
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from api_support import ALEX, AUTH_SECRET, OTHER_TEAM, OUTSIDER, SARAH, TEAM
from fastapi.testclient import TestClient
from test_auth import bearer
from test_google_auth_api import CLIENT_ID, CLIENT_SECRET, REDIRECT_URL, FakeGoogle

from brain.api.deps import get_http_transport, get_settings
from brain.auth import hash_password
from brain.store import NotFound

PASSWORD = "priya-password-1"
OLGA_ADMIN = OUTSIDER.model_copy(update={"is_admin": True})  # the other team's admin


def run(coroutine):
    return asyncio.run(coroutine)


def invite(client: TestClient, email: str = "priya@example.com", **fields) -> httpx.Response:
    body = {"name": "Priya Natarajan", "email": email, "invite": True} | fields
    return client.post("/team/accounts", json=body)


def sign_up(app, email: str = "priya@example.com", password: str = PASSWORD) -> httpx.Response:
    body = {"name": "Priya Natarajan", "email": email, "password": password}
    return TestClient(app).post("/auth/signup", json=body)


def members(client: TestClient) -> dict[str, dict]:
    return {p["email"] or p["id"]: p for p in client.get("/team/members").json()}


# inviting


def test_an_invite_adds_the_person_to_the_admins_team_with_no_login(client_as, store):
    response = invite(client_as(ALEX), email="  Priya@Example.com ", title="Designer")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["password"] is None
    person = body["person"]
    assert (person["name"], person["email"], person["title"]) == (
        "Priya Natarajan",
        "Priya@Example.com",
        "Designer",
    )
    assert person["id"] in run(store.team(TEAM.id)).member_ids
    with pytest.raises(NotFound):
        run(store.login(person["id"]))
    with pytest.raises(NotFound):
        run(store.login_by_email("priya@example.com"))


def test_an_invite_may_make_an_admin(client_as):
    person = invite(client_as(ALEX), is_admin=True).json()["person"]

    assert person["is_admin"] is True


def test_invited_people_are_marked_in_the_members_list_until_they_sign_up(app, client_as):
    alex = client_as(ALEX)
    invite(alex)
    created = alex.post(
        "/team/accounts", json={"name": "Sam Lee", "email": "sam@example.com"}
    ).json()

    listed = members(client_as(SARAH))
    assert listed["priya@example.com"]["invited"] is True
    assert listed["sam@example.com"]["invited"] is False  # has a generated password
    assert listed[ALEX.id]["invited"] is False  # no email: nobody to invite
    assert created["person"]["invited"] is False

    assert sign_up(app).status_code == 201
    assert members(client_as(SARAH))["priya@example.com"]["invited"] is False


def test_only_an_admin_invites(client_as, store):
    for person in (SARAH, OUTSIDER):
        response = invite(client_as(person))

        assert response.status_code == 403, person.id
    assert len(run(store.team(TEAM.id)).member_ids) == 2
    assert run(store.invited_people("priya@example.com")) == []


@pytest.mark.parametrize("email", ["sarah@example.com", "SARAH@Example.com ", "olga@example.com"])
def test_an_email_that_signs_in_already_cant_be_invited(client_as, store, email):
    run(store.set_login(SARAH.id, "sarah@example.com", hash_password("sarah-password-1")))
    run(store.set_login(OUTSIDER.id, "olga@example.com", hash_password("olga-password-1")))

    response = invite(client_as(ALEX), email=email)

    assert response.status_code == 409, response.text
    assert len(run(store.team(TEAM.id)).member_ids) == 2


def test_a_teammates_email_cant_be_invited_twice(client_as, store):
    alex = client_as(ALEX)
    assert invite(alex).status_code == 201

    assert invite(alex, email="PRIYA@example.com").status_code == 409
    assert len(run(store.team(TEAM.id)).member_ids) == 3


@pytest.mark.parametrize(
    "changes",
    [{"email": "not-an-email"}, {"email": ""}, {"name": "   "}, {"name": "x" * 81}],
)
def test_a_bad_invite_is_422(client_as, store, changes):
    response = invite(client_as(ALEX), **changes)

    assert response.status_code == 422, changes
    assert len(run(store.team(TEAM.id)).member_ids) == 2


# signing up


def test_the_invited_person_signs_up_with_a_password_once(app, client_as):
    person = invite(client_as(ALEX)).json()["person"]

    first = sign_up(app, email="PRIYA@example.com")
    second = sign_up(app, password="another-password-2")

    assert first.status_code == 201, first.text
    session = first.json()
    assert session["person"]["id"] == person["id"]
    me = TestClient(app).get("/team", headers=bearer(session["token"]))
    assert me.json()["id"] == TEAM.id
    assert second.status_code == 409
    login = TestClient(app).post(
        "/auth/login", json={"email": "priya@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200, login.text


def test_someone_invited_by_another_team_joins_that_team(app, client_as):
    person = invite(client_as(OLGA_ADMIN), email="sam@example.com").json()["person"]

    response = sign_up(app, email="sam@example.com")

    assert response.status_code == 201, response.text
    token = response.json()["token"]
    assert response.json()["person"]["id"] == person["id"]
    assert TestClient(app).get("/team", headers=bearer(token)).json()["id"] == OTHER_TEAM.id


def test_an_email_nobody_invited_is_403(app, client_as):
    invite(client_as(ALEX))

    response = sign_up(app, email="stranger@example.com")

    assert response.status_code == 403
    assert "invite" in response.json()["detail"].lower()


def test_an_email_invited_by_two_teams_is_409_until_an_admin_sorts_it_out(app, client_as, store):
    first = invite(client_as(ALEX)).json()["person"]
    second = invite(client_as(OLGA_ADMIN))
    assert second.status_code == 201, second.text  # not a login, nor Olga's teammate

    response = sign_up(app)

    assert response.status_code == 409
    assert "Ask your team's admin" in response.json()["detail"]
    for person in (first, second.json()["person"]):
        with pytest.raises(NotFound):
            run(store.login(person["id"]))


# Google


@pytest.fixture
def google(app, settings) -> FakeGoogle:
    fake = FakeGoogle()
    app.dependency_overrides[get_http_transport] = lambda: httpx.MockTransport(fake)
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={
            "auth_secret": AUTH_SECRET,
            "google_client_id": CLIENT_ID,
            "google_client_secret": CLIENT_SECRET,
            "google_redirect_url": REDIRECT_URL,
        }
    )
    return fake


def google_sign_in(app, google: FakeGoogle, email: str) -> dict:
    """The whole round trip; returns the query the browser lands on /login with."""
    browser = TestClient(app, base_url="https://testserver", follow_redirects=False)
    started = browser.get("/auth/google", params={"next": "/"})
    sent = {k: v[0] for k, v in parse_qs(urlsplit(started.headers["location"]).query).items()}
    google.nonce, google.email = sent["nonce"], email
    back = browser.get("/auth/google/callback", params={"code": "c", "state": sent["state"]})
    return {k: v[0] for k, v in parse_qs(urlsplit(back.headers["location"]).query).items()}


def test_an_invited_person_signs_in_with_google_which_reserves_the_login(
    app, client_as, google, store
):
    person = invite(client_as(ALEX)).json()["person"]

    back = google_sign_in(app, google, "priya@example.com")

    session = TestClient(app).post("/auth/google/exchange", json={"code": back["google"]})
    assert session.status_code == 200, session.text
    assert session.json()["person"]["id"] == person["id"]
    run(store.login(person["id"]))  # reserved: nobody can now sign up with a password
    assert sign_up(app).status_code == 409
    assert members(client_as(SARAH))["priya@example.com"]["invited"] is False


def test_google_turns_away_an_email_nobody_invited(app, google):
    assert google_sign_in(app, google, "stranger@example.com") == {"google_error": "not_invited"}


def test_google_refuses_an_email_invited_by_two_teams(app, client_as, google, store):
    invite(client_as(ALEX))
    invite(client_as(OLGA_ADMIN))

    assert google_sign_in(app, google, "priya@example.com") == {"google_error": "ambiguous"}
    assert len(run(store.invited_people("priya@example.com"))) == 2  # no login reserved
