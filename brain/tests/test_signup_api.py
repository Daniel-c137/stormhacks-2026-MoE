"""Creating an account (#128): only an email an admin invited, that is, a person on a team with
that email and no login yet (#143). Sign-up gives them a password login on their own team and
signs them in. test_invites_api.py covers inviting."""

import asyncio

import pytest
from api_support import ALEX, AUTH_SECRET, OTHER_TEAM, OUTSIDER, TEAM
from fastapi.testclient import TestClient
from test_auth import base_settings, bearer

from brain.api.deps import get_login_limiter, get_settings
from brain.auth import LoginLimiter, hash_password, verify_password
from brain.store import NotFound
from contracts import LoginResponse, Person

PRIYA = Person(
    id="u-priya", name="Priya", short="Priya", initials="P", email="Priya.Shah@Example.com"
)  # invited by the admin: on the team, no login yet
PASSWORD = "priya-password-1"


@pytest.fixture
def settings():
    return base_settings(auth_secret=AUTH_SECRET)


@pytest.fixture
def limiter(app) -> LoginLimiter:
    limiter = LoginLimiter()
    app.dependency_overrides[get_login_limiter] = lambda: limiter
    return limiter


@pytest.fixture
def client(app, store, limiter) -> TestClient:
    async def seed():
        await store.upsert_person(PRIYA, TEAM.id)
        await store.upsert_person(
            OUTSIDER.model_copy(update={"email": "olga@elsewhere.com"}), "t-2"
        )
        await store.set_login(ALEX.id, "alex@example.com", hash_password("alex-password-1"))

    asyncio.run(seed())
    return TestClient(app)


def sign_up(client: TestClient, email: str = "priya.shah@example.com", **fields):
    body = {"name": "Priya Shah", "email": email, "password": PASSWORD} | fields
    return client.post("/auth/signup", json=body)


# an invited email


def test_an_invited_email_creates_an_account_and_is_signed_in(client):
    response = sign_up(client)

    assert response.status_code == 201, response.text
    session = LoginResponse.model_validate(response.json())
    assert session.person.id == PRIYA.id
    assert (session.person.name, session.person.short, session.person.initials) == (
        "Priya Shah",
        "Priya",
        "PS",
    )
    me = client.get("/me", headers=bearer(session.token))
    assert me.status_code == 200, me.text


def test_the_new_account_signs_in_with_its_password_afterwards(client):
    sign_up(client, email="  PRIYA.SHAH@example.com ")

    response = client.post(
        "/auth/login", json={"email": "priya.shah@example.com", "password": PASSWORD}
    )

    assert response.status_code == 200, response.text
    assert response.json()["person"]["id"] == PRIYA.id


def test_the_password_is_stored_hashed(client, store):
    sign_up(client)

    login = asyncio.run(store.login(PRIYA.id))
    assert PASSWORD not in login.password_hash
    assert verify_password(login.password_hash, PASSWORD)


# everyone else


def test_an_email_nobody_invited_cannot_create_an_account(client):
    response = sign_up(client, email="stranger@example.com")

    assert response.status_code == 403
    assert "invite" in response.json()["detail"].lower()


def test_someone_invited_by_another_team_signs_up_to_that_team(client):
    response = sign_up(client, email="olga@elsewhere.com")

    assert response.status_code == 201, response.text
    token = response.json()["token"]
    assert response.json()["person"]["id"] == OUTSIDER.id
    assert client.get("/team", headers=bearer(token)).json()["id"] == OTHER_TEAM.id


def test_a_person_without_an_email_cannot_be_signed_up_for(client):
    """Alex and Sarah have no email on the team; only Alex has a login."""
    assert sign_up(client, email="sarah@example.com").status_code == 403


def test_an_email_that_already_has_an_account_is_told_to_sign_in(client):
    first = sign_up(client)
    second = sign_up(client, password="another-password-2")

    assert first.status_code == 201
    assert second.status_code == 409
    assert "sign in" in second.json()["detail"].lower()
    assert sign_up(client, email="alex@example.com").status_code == 409


def stale_reads(store, monkeypatch, person: Person) -> None:
    """The store as a second request saw it before the first one wrote: no login for the email
    yet, and the person still invited."""

    async def no_login(email: str):
        raise NotFound(email)

    async def still_invited(email: str) -> list[Person]:
        return [person]

    monkeypatch.setattr(store, "login_by_email", no_login)
    monkeypatch.setattr(store, "invited_people", still_invited)


def test_a_sign_up_that_loses_a_race_doesnt_replace_the_winners_account(client, store, monkeypatch):
    """Two sign-ups for one invited email can both pass the checks before either writes. Only
    the first gets the account: the second may not replace its password or its name."""
    assert sign_up(client).status_code == 201
    stale_reads(store, monkeypatch, PRIYA)

    second = sign_up(client, name="Mallory", password="mallory-password-9")

    monkeypatch.undo()
    assert second.status_code == 409, second.text
    login = asyncio.run(store.login(PRIYA.id))
    assert verify_password(login.password_hash, PASSWORD)
    assert asyncio.run(store.person(PRIYA.id)).name == "Priya Shah"


def test_signing_up_keeps_the_email_as_the_admin_invited_it(client, store):
    """Priya was invited as Priya.Shah@Example.com and types it in lower case: the team keeps
    the invited email, as Google sign-in does."""
    response = sign_up(client, email="priya.shah@example.com")

    assert response.status_code == 201, response.text
    assert response.json()["person"]["email"] == PRIYA.email
    assert asyncio.run(store.person(PRIYA.id)).email == PRIYA.email
    assert asyncio.run(store.login(PRIYA.id)).email == PRIYA.email


def test_repeated_refusals_lock_the_email(client):
    for _ in range(5):
        assert sign_up(client, email="stranger@example.com").status_code == 403

    response = sign_up(client, email="stranger@example.com")

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"password": "short-pw"}, "password under 10 characters"),
        ({"password": "x" * 1025}, "password over 1024 characters"),
        ({"name": "   "}, "blank name"),
        ({"email": "not-an-email"}, "not an email"),
    ],
)
def test_a_bad_sign_up_is_422_and_creates_nothing(client, store, fields, why):
    response = sign_up(client, **fields)

    assert response.status_code == 422, why
    with pytest.raises(NotFound):
        asyncio.run(store.login(PRIYA.id))


# not set up


def test_sign_up_without_an_auth_secret_is_unavailable(app, client, settings):
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"auth_secret": None}
    )

    assert sign_up(client).status_code == 503


# what the sign-in page offers


def test_the_page_is_told_sign_up_is_open_and_google_is_not_set_up(client):
    assert client.get("/auth/options").json() == {"signup": True, "google": False}


def test_sign_up_is_offered_whenever_sessions_can_be_signed(app, client, settings):
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"auth_secret": None}
    )

    assert client.get("/auth/options").json() == {"signup": False, "google": False}


def test_there_is_no_sign_up_team_setting_any_more():
    assert not hasattr(base_settings(), "signup_team_id")


def test_google_is_offered_once_its_client_and_public_callback_are_configured(
    app, client, settings
):
    google = {
        "google_client_id": "id.apps.googleusercontent.com",
        "google_client_secret": "secret",
    }
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(update=google)
    assert client.get("/auth/options").json()["google"] is False

    callback = {"google_redirect_url": "https://skyroom.example/api/auth/google/callback"}
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(update=google | callback)
    assert client.get("/auth/options").json()["google"] is True
