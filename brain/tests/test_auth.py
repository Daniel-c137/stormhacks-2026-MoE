"""Supabase sessions end to end: a locally signed access token on a real route, no auth override.

Tokens are shaped like Supabase Auth's access tokens (sub, aud "authenticated", role, iss
"<SUPABASE_URL>/auth/v1", exp, iat, session_id). HS256 uses the legacy JWT secret; the
asymmetric path serves an EC P-256 key from a local JWKS endpoint.
"""

import socket
import threading
import time
import uuid

import jwt
import pytest
import uvicorn
from api_support import ALEX, LIVEKIT_URL, TEAM, WORKER_TOKEN
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.auth import InvalidToken, Jwks
from brain.config import Settings
from contracts import Person

JWT_SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"
STRANGER = Person(id=str(uuid.uuid4()), name="Sam Stranger", short="Sam", initials="SS")


def claims(sub: str = ALEX.id, issuer: str | None = None, **overrides) -> dict:
    now = int(time.time())
    body = {
        "sub": sub,
        "aud": "authenticated",
        "role": "authenticated",
        "exp": now + 3600,
        "iat": now,
        "email": "alex@example.com",
        "session_id": str(uuid.uuid4()),
        "is_anonymous": False,
    }
    if issuer:
        body["iss"] = issuer
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not None}


def hs256(secret: str = JWT_SECRET, **kw) -> str:
    return jwt.encode(claims(**kw), secret, algorithm="HS256")


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def base_settings(**kw) -> Settings:
    return Settings(
        _env_file=None,
        livekit_url=LIVEKIT_URL,
        livekit_api_key="test-key",
        livekit_api_secret="test-secret-that-is-long-enough-for-hs256",
        brain_internal_token=WORKER_TOKEN,
        **kw,
    )


@pytest.fixture
def settings() -> Settings:
    return base_settings(supabase_jwt_secret=JWT_SECRET)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# HS256 (legacy JWT secret)


def test_a_valid_session_resolves_to_the_member_on_a_real_route(client):
    response = client.post("/meetings", json={"title": "Standup"}, headers=bearer(hs256()))

    assert response.status_code == 200, response.text
    assert response.json()["host_id"] == ALEX.id
    assert response.json()["team_id"] == TEAM.id
    listed = client.get("/meetings", headers=bearer(hs256()))
    assert [m["title"] for m in listed.json()] == ["Standup"]


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": ""}, {"Authorization": "Bearer"}, {"Authorization": "Basic abc"}],
    ids=["missing", "empty", "no-token", "wrong-scheme"],
)
def test_a_missing_or_malformed_header_is_401_with_a_bearer_challenge(client, headers):
    response = client.get("/meetings", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


@pytest.mark.parametrize(
    "token",
    [
        hs256(secret="a-different-secret-that-is-also-long-enough"),
        hs256(exp=int(time.time()) - 120),
        hs256(nbf=int(time.time()) + 120),
        hs256(aud="anon"),
        hs256(aud=None),
        hs256(exp=None),
        "not-a-jwt",
        jwt.encode(claims(), key=None, algorithm="none"),
    ],
    ids=[
        "bad-signature",
        "expired",
        "not-yet-valid",
        "wrong-aud",
        "no-aud",
        "no-exp",
        "junk",
        "none",
    ],
)
def test_an_invalid_token_is_401(client, token):
    response = client.get("/meetings", headers=bearer(token))

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_a_token_just_past_expiry_is_accepted_within_the_leeway(client):
    response = client.get("/meetings", headers=bearer(hs256(exp=int(time.time()) - 3)))

    assert response.status_code == 200, response.text


def test_the_audience_is_configurable(app, client):
    from brain.api.deps import get_settings

    custom = base_settings(supabase_jwt_secret=JWT_SECRET, supabase_jwt_audience="omni")
    app.dependency_overrides[get_settings] = lambda: custom

    assert client.get("/meetings", headers=bearer(hs256(aud="omni"))).status_code == 200
    assert client.get("/meetings", headers=bearer(hs256())).status_code == 401


def test_a_valid_token_for_someone_outside_the_workspace_is_403(client):
    response = client.get("/meetings", headers=bearer(hs256(sub=STRANGER.id)))

    assert response.status_code == 403
    assert response.json()["detail"] == "not a workspace member"


def test_no_supabase_config_is_503_never_an_anonymous_user(app, client):
    from brain.api.deps import get_settings

    app.dependency_overrides[get_settings] = lambda: base_settings()

    for headers in ({}, bearer(hs256())):
        response = client.get("/meetings", headers=headers)
        assert response.status_code == 503
        assert response.json()["detail"] == "Supabase auth is not configured"


def test_the_issuer_is_checked_when_supabase_url_is_set(app, client):
    from brain.api.deps import get_settings

    url = "https://abcd.supabase.co"
    app.dependency_overrides[get_settings] = lambda: base_settings(
        supabase_jwt_secret=JWT_SECRET, supabase_url=url
    )

    ok = client.get("/meetings", headers=bearer(hs256(issuer=f"{url}/auth/v1")))
    wrong = client.get(
        "/meetings", headers=bearer(hs256(issuer="https://evil.supabase.co/auth/v1"))
    )
    missing = client.get("/meetings", headers=bearer(hs256()))

    assert ok.status_code == 200, ok.text
    assert wrong.status_code == 401
    assert missing.status_code == 401


# Asymmetric signing keys (JWKS)


class SupabaseKeys:
    """A project's signing keys served at <url>/auth/v1/.well-known/jwks.json on localhost."""

    def __init__(self):
        self.keys: dict[str, ec.EllipticCurvePrivateKey] = {}
        self.fetches = 0
        self.url = ""
        self.rotate()
        app = FastAPI()

        @app.get("/auth/v1/.well-known/jwks.json")
        def jwks() -> dict:
            self.fetches += 1
            return {"keys": [self.public_jwk(kid) for kid in self.keys]}

        self.app = app

    def rotate(self) -> str:
        kid = str(uuid.uuid4())
        self.keys[kid] = ec.generate_private_key(ec.SECP256R1())
        return kid

    def public_jwk(self, kid: str) -> dict:
        jwk = jwt.algorithms.ECAlgorithm.to_jwk(self.keys[kid].public_key(), as_dict=True)
        return {**jwk, "kid": kid, "alg": "ES256", "use": "sig", "key_ops": ["verify"]}

    def sign(self, kid: str | None = None, **kw) -> str:
        kid = kid or next(iter(self.keys))
        return jwt.encode(
            claims(issuer=f"{self.url}/auth/v1", **kw),
            self.keys[kid],
            algorithm="ES256",
            headers={"kid": kid},
        )


@pytest.fixture
def supabase():
    keys = SupabaseKeys()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(keys.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("fake JWKS server did not start")
        time.sleep(0.02)
    keys.url = f"http://127.0.0.1:{port}"
    yield keys
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def jwks_client(app, supabase) -> TestClient:
    """Only SUPABASE_URL is configured: tokens verify against the project's JWKS."""
    from brain.api.deps import get_settings

    app.dependency_overrides[get_settings] = lambda: base_settings(supabase_url=supabase.url)
    return TestClient(app)


def test_an_es256_token_verifies_against_the_project_jwks(jwks_client, supabase):
    response = jwks_client.post(
        "/meetings", json={"title": "Retro"}, headers=bearer(supabase.sign())
    )

    assert response.status_code == 200, response.text
    assert response.json()["host_id"] == ALEX.id


def test_the_jwks_is_cached_between_requests(jwks_client, supabase):
    for _ in range(3):
        assert jwks_client.get("/meetings", headers=bearer(supabase.sign())).status_code == 200

    assert supabase.fetches == 1


def test_an_unknown_kid_refetches_the_jwks_once(jwks_client, supabase):
    assert jwks_client.get("/meetings", headers=bearer(supabase.sign())).status_code == 200
    rotated = supabase.rotate()

    response = jwks_client.get("/meetings", headers=bearer(supabase.sign(kid=rotated)))

    assert response.status_code == 200, response.text
    assert supabase.fetches == 2


def test_a_kid_the_project_never_published_is_401_after_one_refetch(jwks_client, supabase):
    assert jwks_client.get("/meetings", headers=bearer(supabase.sign())).status_code == 200
    token = jwt.encode(
        claims(issuer=f"{supabase.url}/auth/v1"),
        ec.generate_private_key(ec.SECP256R1()),
        algorithm="ES256",
        headers={"kid": "not-published"},
    )

    response = jwks_client.get("/meetings", headers=bearer(token))

    assert response.status_code == 401
    assert supabase.fetches == 2  # the first load, then exactly one refetch


def test_a_token_signed_by_another_key_under_a_published_kid_is_401(jwks_client, supabase):
    kid = next(iter(supabase.keys))
    forged = jwt.encode(
        claims(issuer=f"{supabase.url}/auth/v1"),
        ec.generate_private_key(ec.SECP256R1()),
        algorithm="ES256",
        headers={"kid": kid},
    )

    assert jwks_client.get("/meetings", headers=bearer(forged)).status_code == 401


def test_hs256_is_refused_when_only_the_jwks_is_configured(jwks_client, supabase):
    token = hs256(issuer=f"{supabase.url}/auth/v1")

    assert jwks_client.get("/meetings", headers=bearer(token)).status_code == 401


def test_an_expired_or_wrong_audience_es256_token_is_401(jwks_client, supabase):
    expired = supabase.sign(exp=int(time.time()) - 120)
    wrong_aud = supabase.sign(aud="anon")

    assert jwks_client.get("/meetings", headers=bearer(expired)).status_code == 401
    assert jwks_client.get("/meetings", headers=bearer(wrong_aud)).status_code == 401


def test_an_unreachable_jwks_is_503(app):
    from brain.api.deps import get_settings

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app.dependency_overrides[get_settings] = lambda: base_settings(
        supabase_url=f"http://127.0.0.1:{port}"
    )
    keys = SupabaseKeys()
    keys.url = f"http://127.0.0.1:{port}"

    response = TestClient(app).get("/meetings", headers=bearer(keys.sign()))

    assert response.status_code == 503


# The key cache on its own


@pytest.mark.anyio
async def test_the_jwks_cache_expires_after_its_ttl():
    keys = SupabaseKeys()
    kid = next(iter(keys.keys))
    calls = []
    now = [1000.0]

    async def fetch(url: str) -> dict:
        calls.append(url)
        return {"keys": [keys.public_jwk(k) for k in keys.keys]}

    cache = Jwks(
        "https://abcd.supabase.co/auth/v1/.well-known/jwks.json",
        fetch=fetch,
        ttl=600,
        clock=lambda: now[0],
    )

    await cache.key(kid)
    now[0] += 599
    await cache.key(kid)
    assert len(calls) == 1

    now[0] += 2
    await cache.key(kid)
    assert len(calls) == 2


@pytest.mark.anyio
async def test_an_unknown_kid_refetches_once_but_not_right_after_a_load():
    calls = []
    now = [1000.0]

    async def fetch(url: str) -> dict:
        calls.append(url)
        return {"keys": []}

    cache = Jwks(
        "https://abcd.supabase.co/auth/v1/.well-known/jwks.json",
        fetch=fetch,
        clock=lambda: now[0],
    )

    with pytest.raises(InvalidToken):
        await cache.key("missing")
    assert len(calls) == 1  # the keys were just loaded; fetching again would not help

    now[0] += 60
    with pytest.raises(InvalidToken):
        await cache.key("missing")
    assert len(calls) == 2  # still fresh, so the unknown kid is what triggers one refetch
