"""Google sign-in and sign-up through the brain (#128), with no third-party auth service.

GET /auth/google sends the browser to Google with state, a nonce and a PKCE challenge, kept in a
signed, short-lived cookie. Google sends it back to /auth/google/callback with a code; the brain
swaps the code for an ID token, verifies it against Google's published keys, and signs in the
account with that email, or an invited email (whose login is reserved for Google). The browser
returns to the board with a one-time code, never the session token, which the board exchanges at
POST /auth/google/exchange. Google is faked here: its token endpoint and its keys.
"""

import asyncio
import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
from api_support import ALEX, AUTH_SECRET, TEAM
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from test_auth import base_settings, bearer

from brain.api.deps import get_http_transport, get_settings
from brain.auth import hash_password
from brain.store import NotFound
from contracts import LoginResponse, Person

CLIENT_ID = "client-123.apps.googleusercontent.com"
CLIENT_SECRET = "google-client-secret"
BOARD = "https://board.test"
PRIYA = Person(
    id="u-priya", name="Priya Shah", short="Priya", initials="PS", email="priya@example.com"
)


class FakeGoogle:
    """Google's token endpoint and published keys. The test sets who the next code signs in as;
    `tamper` changes the ID token's claims; `other_key` signs it with a key Google never
    published."""

    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.stranger_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.email = "alex@example.com"
        self.tamper: dict = {}
        self.other_key = False
        self.token_requests: list[dict] = []
        self.nonce: str | None = None

    def id_token(self) -> str:
        now = int(time.time())
        claims = {
            "iss": "https://accounts.google.com",
            "aud": CLIENT_ID,
            "sub": "google-sub-1",
            "email": self.email,
            "email_verified": True,
            "name": "From Google",
            "iat": now,
            "exp": now + 600,
            "nonce": self.nonce,
        } | self.tamper
        key = self.stranger_key if self.other_key else self.key
        return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "key-1"})

    def jwks(self) -> dict:
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        return {"keys": [jwk | {"kid": "key-1", "use": "sig", "alg": "RS256"}]}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com" and request.url.path == "/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.token_requests.append(form)
            return httpx.Response(200, json={"id_token": self.id_token(), "token_type": "Bearer"})
        if request.url.host == "www.googleapis.com" and request.url.path == "/oauth2/v3/certs":
            return httpx.Response(200, json=self.jwks())
        return httpx.Response(404)


@pytest.fixture
def settings():
    return base_settings(
        auth_secret=AUTH_SECRET,
        signup_team_id=TEAM.id,
        google_client_id=CLIENT_ID,
        google_client_secret=CLIENT_SECRET,
        board_url=BOARD,
    )


@pytest.fixture
def google(app) -> FakeGoogle:
    fake = FakeGoogle()
    app.dependency_overrides[get_http_transport] = lambda: httpx.MockTransport(fake)
    return fake


@pytest.fixture
def client(app, store, google) -> TestClient:
    async def seed():
        await store.set_login(ALEX.id, "Alex@Example.com", hash_password("alex-password-1"))
        await store.upsert_person(PRIYA, TEAM.id)  # invited, no login yet

    asyncio.run(seed())
    return TestClient(app, follow_redirects=False)


def start(client: TestClient, google: FakeGoogle, next_path: str = "/meetings") -> dict:
    """GET /auth/google; returns the query Google was sent, and remembers its nonce."""
    response = client.get("/auth/google", params={"next": next_path})
    assert response.status_code in (302, 307), response.text
    sent = {k: v[0] for k, v in parse_qs(urlsplit(response.headers["location"]).query).items()}
    google.nonce = sent.get("nonce")
    return sent


def back_from_google(client: TestClient, **params) -> httpx.Response:
    return client.get("/auth/google/callback", params=params)


def landed(response: httpx.Response) -> dict:
    """Where the callback sent the browser: the board's /login, and its query."""
    assert response.status_code in (302, 307), response.text
    url = urlsplit(response.headers["location"])
    assert f"{url.scheme}://{url.netloc}{url.path}" == f"{BOARD}/login"
    return {k: v[0] for k, v in parse_qs(url.query).items()}


def exchange(client: TestClient, code: str) -> httpx.Response:
    return client.post("/auth/google/exchange", json={"code": code})


def sign_in_with_google(client: TestClient, google: FakeGoogle, next_path: str = "/meetings"):
    sent = start(client, google, next_path)
    return landed(back_from_google(client, code="google-code", state=sent["state"]))


# going to Google


def test_the_browser_is_sent_to_google_with_state_nonce_and_pkce(client, google):
    response = client.get("/auth/google")

    url = urlsplit(response.headers["location"])
    assert (url.scheme, url.netloc, url.path) == (
        "https",
        "accounts.google.com",
        "/o/oauth2/v2/auth",
    )
    sent = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert sent["client_id"] == CLIENT_ID
    assert sent["response_type"] == "code"
    assert set(sent["scope"].split()) >= {"openid", "email", "profile"}
    assert sent["redirect_uri"].endswith("/auth/google/callback")
    assert sent["code_challenge_method"] == "S256"
    assert len(sent["state"]) >= 32 and len(sent["nonce"]) >= 32
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert CLIENT_SECRET not in response.headers["location"]


def test_google_isnt_offered_until_its_client_is_configured(app, client, settings):
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"google_client_secret": None}
    )

    assert client.get("/auth/google").status_code == 503


# coming back


def test_an_existing_account_signs_in_with_google(client, google):
    back = sign_in_with_google(client, google)

    assert back["next"] == "/meetings"
    response = exchange(client, back["google"])
    assert response.status_code == 200, response.text
    session = LoginResponse.model_validate(response.json())
    assert session.person.id == ALEX.id
    assert client.get("/me", headers=bearer(session.token)).status_code == 200


def test_the_code_is_swapped_with_the_pkce_verifier_and_client_secret(client, google):
    sent = start(client, google)
    back_from_google(client, code="google-code", state=sent["state"])

    [form] = google.token_requests
    assert form["code"] == "google-code"
    assert form["grant_type"] == "authorization_code"
    assert (form["client_id"], form["client_secret"]) == (CLIENT_ID, CLIENT_SECRET)
    assert form["redirect_uri"] == sent["redirect_uri"]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest())
    assert challenge.rstrip(b"=").decode() == sent["code_challenge"]


def test_an_invited_email_creates_its_account_with_google(client, google, store):
    google.email = "PRIYA@example.com"

    session = exchange(client, sign_in_with_google(client, google)["google"]).json()

    assert session["person"]["id"] == PRIYA.id
    asyncio.run(store.login(PRIYA.id))  # reserved, so nobody can sign up as Priya with a password
    taken = client.post(
        "/auth/signup",
        json={"name": "Not Priya", "email": "priya@example.com", "password": "x" * 12},
    )
    assert taken.status_code == 409


def test_a_google_account_nobody_invited_is_turned_away(client, google, store):
    google.email = "stranger@example.com"

    back = sign_in_with_google(client, google)

    assert back == {"google_error": "not_invited"}


# what is never trusted


def test_a_callback_without_the_matching_state_is_refused(client, google):
    start(client, google)

    back = landed(back_from_google(client, code="google-code", state="forged-state"))

    assert back == {"google_error": "state"}
    assert google.token_requests == []


def test_a_callback_from_another_browser_is_refused(app, google):
    """Login CSRF: the state only works in the browser that started the sign-in."""
    starter = TestClient(app, follow_redirects=False)
    sent = start(starter, google)
    victim = TestClient(app, follow_redirects=False)

    back = landed(back_from_google(victim, code="attacker-code", state=sent["state"]))

    assert back == {"google_error": "state"}


def test_cancelling_at_google_returns_to_the_sign_in_page(client, google):
    sent = start(client, google)

    back = landed(back_from_google(client, error="access_denied", state=sent["state"]))

    assert back == {"google_error": "cancelled"}


@pytest.mark.parametrize(
    ("tamper", "other_key", "reason"),
    [
        ({"aud": "someone-elses-client"}, False, "failed"),
        ({"iss": "https://evil.example"}, False, "failed"),
        ({"exp": int(time.time()) - 3600}, False, "failed"),
        ({"nonce": "replayed-nonce"}, False, "failed"),
        ({}, True, "failed"),
        ({"email_verified": False}, False, "unverified"),
    ],
    ids=["audience", "issuer", "expired", "nonce", "unpublished-key", "unverified-email"],
)
def test_an_id_token_that_doesnt_check_out_signs_nobody_in(
    client, google, tamper, other_key, reason
):
    google.tamper, google.other_key = tamper, other_key

    back = sign_in_with_google(client, google)

    assert back == {"google_error": reason}


def test_the_one_time_code_works_once(client, google):
    code = sign_in_with_google(client, google)["google"]

    assert exchange(client, code).status_code == 200
    assert exchange(client, code).status_code == 401
    assert exchange(client, "made-up").status_code == 401


def test_only_a_same_site_path_is_kept_as_where_to_go_next(client, google):
    assert sign_in_with_google(client, google, next_path="//evil.example")["next"] == "/"
    assert sign_in_with_google(client, google, next_path="https://evil.example")["next"] == "/"


def test_a_failed_google_sign_in_leaves_no_login_behind(client, google, store):
    google.email, google.tamper = "priya@example.com", {"aud": "someone-elses-client"}

    sign_in_with_google(client, google)

    with pytest.raises(NotFound):
        asyncio.run(store.login(PRIYA.id))
