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
from test_google_auth_api import CLIENT_ID, CLIENT_SECRET, REDIRECT_URL, FakeGoogle

from brain.api.deps import get_http_transport, get_settings
from brain.auth import hash_password
from brain.store import NotFound
from contracts import Person

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
    [{"email": "not-an-email"}, {"email": ""}, {"name": "x" * 81}],
)
def test_a_bad_invite_is_422(client_as, store, changes):
    response = invite(client_as(ALEX), **changes)

    assert response.status_code == 422, changes
    assert len(run(store.team(TEAM.id)).member_ids) == 2


@pytest.mark.parametrize(
    ("email", "name"),
    [
        ("priya.natarajan@example.com", "Priya Natarajan"),
        ("priya@example.com", "Priya"),
        ("sam_lee-2@example.com", "Sam Lee 2"),
    ],
)
@pytest.mark.parametrize(
    "given", [{}, {"name": None}, {"name": "   "}], ids=["none", "null", "blank"]
)
def test_an_invite_needs_only_an_email(client_as, store, email, name, given):
    """The admin adds only the email; until the person signs up they show under a name made from
    it, and they choose their own name and password when they create the account."""
    body = {"email": email, "invite": True} | given
    response = client_as(ALEX).post("/team/accounts", json=body)

    assert response.status_code == 201, response.text
    person = response.json()["person"]
    assert (person["name"], person["email"], person["invited"]) == (name, email, True)
    assert response.json()["password"] is None
    with pytest.raises(NotFound):
        run(store.login(person["id"]))


def test_an_account_with_a_generated_password_still_needs_a_name(client_as, store):
    response = client_as(ALEX).post("/team/accounts", json={"email": "sam@example.com"})

    assert response.status_code == 422
    assert len(run(store.team(TEAM.id)).member_ids) == 2


# signing up


def test_signing_up_after_an_email_only_invite_sets_the_name_they_choose(app, client_as, store):
    person = (
        client_as(ALEX)
        .post("/team/accounts", json={"email": "priya@example.com", "invite": True})
        .json()["person"]
    )

    response = sign_up(app)

    assert response.status_code == 201, response.text
    signed_up = run(store.person(person["id"]))
    assert (signed_up.name, signed_up.short, signed_up.initials) == (
        "Priya Natarajan",
        "Priya",
        "PN",
    )


def test_the_invited_person_signs_up_with_a_password_once(app, client_as, store):
    person = invite(client_as(ALEX)).json()["person"]

    first = sign_up(app, email="PRIYA@example.com")
    second = sign_up(app, password="another-password-2")

    assert first.status_code == 201, first.text
    assert first.json()["person"]["id"] == person["id"]
    assert run(store.team_for_user(person["id"])).id == TEAM.id
    assert second.status_code == 409
    login = TestClient(app).post(
        "/auth/login", json={"email": "priya@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200, login.text


def test_someone_invited_by_another_team_joins_that_team(app, client_as, store):
    person = invite(client_as(OLGA_ADMIN), email="sam@example.com").json()["person"]

    response = sign_up(app, email="sam@example.com")

    assert response.status_code == 201, response.text
    assert response.json()["person"]["id"] == person["id"]
    assert run(store.team_for_user(person["id"])).id == OTHER_TEAM.id


def test_an_email_nobody_invited_is_403(app, client_as):
    invite(client_as(ALEX))

    response = sign_up(app, email="stranger@example.com")

    assert response.status_code == 403
    assert "invite" in response.json()["detail"].lower()


def test_an_email_another_team_invited_cant_be_invited_or_given_an_account(app, client_as, store):
    """Two teams inviting one email would leave it unable to sign up, and nobody can remove a
    member yet; giving it a password on a second team would strand the first invite. Both are
    refused, and the first team's invite still works."""
    first = invite(client_as(OLGA_ADMIN)).json()["person"]
    alex = client_as(ALEX)

    again = invite(alex, email="PRIYA@example.com")
    with_password = alex.post(
        "/team/accounts", json={"name": "Priya Natarajan", "email": "priya@example.com"}
    )

    assert again.status_code == 409, again.text
    assert with_password.status_code == 409, with_password.text
    assert "another team" in again.json()["detail"].lower()
    assert len(run(store.team(TEAM.id)).member_ids) == 2
    response = sign_up(app)
    assert response.status_code == 201, response.text
    assert response.json()["person"]["id"] == first["id"]


def two_invites(store) -> list[Person]:
    """The same email on both teams with no login, as a race between two invites leaves it."""
    people = [
        Person(id=f"u-priya-{team.id}", name="Priya", short="Priya", initials="P", email=email)
        for team, email in ((TEAM, "priya@example.com"), (OTHER_TEAM, "PRIYA@example.com"))
    ]
    for person, team in zip(people, (TEAM, OTHER_TEAM), strict=True):
        run(store.upsert_person(person, team.id))
    return people


def test_an_email_invited_by_two_teams_is_409_until_an_admin_sorts_it_out(app, store):
    people = two_invites(store)

    response = sign_up(app)

    assert response.status_code == 409
    assert "Ask your team's admin" in response.json()["detail"]
    for person in people:
        with pytest.raises(NotFound):
            run(store.login(person.id))


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


def google_browser(app) -> TestClient:
    return TestClient(app, base_url="https://testserver", follow_redirects=False)


def google_sign_in(app, google: FakeGoogle, email: str, browser: TestClient | None = None) -> dict:
    """The whole round trip; returns the query the browser lands on /login with."""
    browser = browser or google_browser(app)
    started = browser.get("/auth/google", params={"next": "/"})
    sent = {k: v[0] for k, v in parse_qs(urlsplit(started.headers["location"]).query).items()}
    google.nonce, google.email = sent["nonce"], email
    back = browser.get("/auth/google/callback", params={"code": "c", "state": sent["state"]})
    return {k: v[0] for k, v in parse_qs(urlsplit(back.headers["location"]).query).items()}


def test_an_invited_person_signs_in_with_google_which_reserves_the_login(
    app, client_as, google, store
):
    person = invite(client_as(ALEX)).json()["person"]

    browser = google_browser(app)
    back = google_sign_in(app, google, "priya@example.com", browser)

    session = browser.post("/auth/google/exchange", json={"code": back["google"]})
    assert session.status_code == 200, session.text
    assert session.json()["person"]["id"] == person["id"]
    run(store.login(person["id"]))  # reserved: nobody can now sign up with a password
    assert sign_up(app).status_code == 409
    assert members(client_as(SARAH))["priya@example.com"]["invited"] is False


def test_google_turns_away_an_email_nobody_invited(app, google):
    assert google_sign_in(app, google, "stranger@example.com") == {"google_error": "not_invited"}


def test_google_refuses_an_email_invited_by_two_teams(app, google, store):
    two_invites(store)

    assert google_sign_in(app, google, "priya@example.com") == {"google_error": "ambiguous"}
    assert len(run(store.invited_people("priya@example.com"))) == 2  # no login reserved


def test_google_names_someone_invited_by_email_only(app, client_as, google, store):
    person = (
        client_as(ALEX)
        .post("/team/accounts", json={"email": "priya@example.com", "invite": True})
        .json()["person"]
    )

    browser = google_browser(app)
    back = google_sign_in(app, google, "priya@example.com", browser)
    session = browser.post("/auth/google/exchange", json={"code": back["google"]})

    assert session.status_code == 200, session.text
    assert run(store.person(person["id"])).name == "From Google"


def test_google_keeps_the_name_an_admin_gave(app, client_as, google, store):
    person = invite(client_as(ALEX)).json()["person"]

    browser = google_browser(app)
    back = google_sign_in(app, google, "priya@example.com", browser)
    browser.post("/auth/google/exchange", json={"code": back["google"]})

    assert run(store.person(person["id"])).name == "Priya Natarajan"
