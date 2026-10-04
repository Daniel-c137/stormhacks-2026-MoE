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
from api_support import ALEX, AUTH_SECRET, SARAH, TEAM
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from test_auth import base_settings, bearer
from test_signup_api import stale_reads

from brain.api.deps import get_http_transport, get_settings
from brain.auth import hash_password, verify_password
from brain.google_auth import safe_next
from brain.store import NotFound
from contracts import LoginResponse, Person

CLIENT_ID = "client-123.apps.googleusercontent.com"
CLIENT_SECRET = "google-client-secret"
BOARD = "https://board.test"
# Registered with Google; the brain is served at /api behind the board's origin (Caddy, or the
# board's own /api rewrite locally) and sees only its internal http://127.0.0.1:8000.
REDIRECT_URL = f"{BOARD}/api/auth/google/callback"
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
            "hd": self.email.rpartition("@")[2].lower(),  # a Google Workspace domain
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
        google_client_id=CLIENT_ID,
        google_client_secret=CLIENT_SECRET,
        google_redirect_url=REDIRECT_URL,
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
    return browser(app)


def browser(app, base_url: str = "https://testserver") -> TestClient:
    """A browser on HTTPS, so it returns the Secure flow cookie as a real one would."""
    return TestClient(app, base_url=base_url, follow_redirects=False)


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


def test_google_racing_a_password_sign_up_doesnt_replace_its_password(
    client, google, store, monkeypatch
):
    """Priya signs up with a password while a Google sign-in for her email is past its checks:
    reserving her login for Google must not replace the password she just set."""
    password = "priya-password-1"
    response = client.post(
        "/auth/signup",
        json={"name": "Priya Shah", "email": "priya@example.com", "password": password},
    )
    assert response.status_code == 201, response.text
    stale_reads(store, monkeypatch, PRIYA)
    google.email = "priya@example.com"

    sign_in_with_google(client, google)

    monkeypatch.undo()
    assert verify_password(asyncio.run(store.login(PRIYA.id)).password_hash, password)


@pytest.mark.parametrize("hd", [None, ""], ids=["no-hd", "empty-hd"])
def test_google_signs_in_only_an_email_google_hosts(client, google, store, hd):
    """Google is only authoritative for Gmail and Workspace accounts (`hd` set). Anyone can make a
    Google account with another provider's address, and it stays verified after that address
    changes hands, so it neither signs in an existing account nor takes an invite."""
    google.tamper = {"hd": hd}

    assert sign_in_with_google(client, google) == {"google_error": "unverified"}  # Alex
    google.email = "priya@example.com"
    assert sign_in_with_google(client, google) == {"google_error": "unverified"}
    with pytest.raises(NotFound):
        asyncio.run(store.login(PRIYA.id))


def test_a_gmail_address_signs_in_without_a_workspace_domain(client, google, store):
    asyncio.run(store.set_login(SARAH.id, "sarah.kim@gmail.com", hash_password("sarah-pw-123")))
    google.email, google.tamper = "Sarah.Kim@gmail.com", {"hd": None}

    session = exchange(client, sign_in_with_google(client, google)["google"])

    assert session.status_code == 200, session.text
    assert session.json()["person"]["id"] == SARAH.id


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
    starter = browser(app)
    sent = start(starter, google)
    victim = browser(app)

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


def test_the_one_time_code_only_works_in_the_browser_that_signed_in(app, client, google):
    """Login CSRF: someone who signs in with their own Google account mustn't be able to send
    another person a link to /login?google=<code> that signs that person's browser in as them."""
    code = sign_in_with_google(client, google)["google"]

    assert exchange(browser(app), code).status_code == 401
    assert exchange(client, code).status_code == 401  # and the code is spent


def test_the_code_is_bound_by_an_httponly_lax_cookie_that_the_exchange_clears(client, google):
    sent = start(client, google)
    back = back_from_google(client, code="google-code", state=sent["state"])
    code = landed(back)["google"]

    cookies = [c for c in back.headers.get_list("set-cookie") if c.startswith("google_handoff=")]
    assert len(cookies) == 1, back.headers.get_list("set-cookie")
    attributes = cookies[0].lower()
    assert "httponly" in attributes and "samesite=lax" in attributes
    assert "secure" in attributes and "path=/" in attributes
    assert code not in cookies[0]
    swapped = exchange(client, code)
    assert swapped.status_code == 200, swapped.text
    assert 'google_handoff=""' in swapped.headers["set-cookie"]


def test_only_a_same_site_path_is_kept_as_where_to_go_next(client, google):
    assert sign_in_with_google(client, google, next_path="//evil.example")["next"] == "/"
    assert sign_in_with_google(client, google, next_path="https://evil.example")["next"] == "/"


@pytest.mark.parametrize(
    "next_path",
    [
        "/\\evil.com",  # browsers read a backslash as a slash: //evil.com
        "/\t/evil.com",  # browsers drop tabs and newlines: //evil.com
        "/\n/evil.com",
        "/\x00/evil.com",
        "/\x7f/evil.com",
        "//evil.com",
        "https://evil.com",
        "http:/evil.com",
        "/%5Cevil.com",  # a backslash once decoded
        "/%09/evil.com",
        "evil.com",
        "",
        None,
    ],
)
def test_a_next_that_could_leave_the_site_is_home(next_path):
    assert safe_next(next_path) == "/"


@pytest.mark.parametrize(
    "next_path", ["/", "/meetings/x?y=1#z", "/m/abc-def", "/login?mode=signup"]
)
def test_a_path_on_this_site_is_kept(next_path):
    assert safe_next(next_path) == next_path


@pytest.mark.parametrize("next_path", ["/\\evil.com", "/\t/evil.com", "//evil.com"])
def test_a_bad_next_comes_back_from_google_as_home(client, google, next_path):
    back = sign_in_with_google(client, google, next_path=next_path)

    assert back["next"] == "/"
    assert exchange(client, back["google"]).status_code == 200


def test_an_encoded_backslash_in_the_query_comes_back_as_home(client, google):
    """/login?next=/%5Cevil.com: the board passes the decoded /\\evil.com on to the brain."""
    response = client.get("/auth/google?next=/%5Cevil.com")
    sent = {k: v[0] for k, v in parse_qs(urlsplit(response.headers["location"]).query).items()}
    google.nonce = sent["nonce"]

    back = landed(back_from_google(client, code="google-code", state=sent["state"]))

    assert back["next"] == "/"


def test_a_failed_google_sign_in_leaves_no_login_behind(client, google, store):
    google.email, google.tamper = "priya@example.com", {"aud": "someone-elses-client"}

    sign_in_with_google(client, google)

    with pytest.raises(NotFound):
        asyncio.run(store.login(PRIYA.id))


def test_behind_a_proxy_google_uses_the_public_callback_url_and_the_cookie_still_returns(
    client, google
):
    """In production the brain sits under the board's origin at /api: Google must send the
    browser to the public URL, and the flow cookie must come back on that path too."""
    sent = start(client, google)
    cookie = client.cookies.jar
    back = landed(back_from_google(client, code="google-code", state=sent["state"]))

    assert sent["redirect_uri"] == REDIRECT_URL
    assert google.token_requests[0]["redirect_uri"] == REDIRECT_URL
    assert all(c.path == "/" for c in cookie if c.name == "google_signin")
    assert "google" in back


def test_google_is_off_without_its_public_callback_url(app, client, google, settings):
    """The brain can't build the public URL itself behind the proxy (it sees its internal
    address), so without GOOGLE_REDIRECT_URL Google isn't offered at all."""
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"google_redirect_url": None}
    )

    assert client.get("/auth/options").json()["google"] is False
    assert client.get("/auth/google").status_code == 503
    assert landed(back_from_google(client, code="google-code", state="s")) == {
        "google_error": "failed"
    }
    assert google.token_requests == []


@pytest.mark.parametrize("url", ["not a url", "/api/auth/google/callback", "ftp://x/cb"])
def test_a_callback_url_that_isnt_an_http_url_leaves_google_off(app, client, settings, url):
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"google_redirect_url": url}
    )

    assert client.get("/auth/options").json()["google"] is False
    assert client.get("/auth/google").status_code == 503


def test_the_flow_cookie_is_secure_when_the_callback_is_https(app):
    """Behind the proxy the brain is reached over plain http; the public URL decides."""
    response = browser(app, base_url="http://127.0.0.1:8000").get("/auth/google")

    assert response.status_code == 302, response.text
    assert "secure" in response.headers["set-cookie"].lower()


def test_locally_over_http_the_cookie_isnt_secure_and_sign_in_works(app, client, google, settings):
    local = "http://localhost:3000/api/auth/google/callback"
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"google_redirect_url": local, "board_url": None}
    )
    client = browser(app, base_url="http://localhost:3000")  # seeded by the client fixture

    sent = start(client, google)
    response = back_from_google(client, code="google-code", state=sent["state"])

    assert sent["redirect_uri"] == local
    assert "secure" not in client.get("/auth/google").headers["set-cookie"].lower()
    assert response.headers["location"].startswith("/login?google=")


def test_without_board_url_google_returns_to_the_same_sites_sign_in_page(
    app, client, google, settings
):
    """Production serves the board and the brain on one domain (Caddy, /api), so with no
    BOARD_URL the callback sends the browser to /login on that domain, never to localhost."""
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(update={"board_url": None})

    sent = start(client, google)
    response = back_from_google(client, code="google-code", state=sent["state"])

    location = response.headers["location"]
    assert location.startswith("/login?google=")
    assert "localhost" not in location


def test_the_board_url_is_unset_by_default(settings):
    assert base_settings().board_url is None
