"""The brain's own sessions: argon2 password hashes, and HS256 session tokens it issues and
accepts on real routes with no auth override."""

import time
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from api_support import ALEX, AUTH_SECRET, LIVEKIT_URL, TEAM, WORKER_TOKEN
from fastapi.testclient import TestClient
from pydantic import ValidationError

from brain.auth import (
    AUDIENCE,
    ISSUER,
    AuthNotConfigured,
    InvalidToken,
    TokenVerifier,
    hash_password,
    issue_token,
    verify_password,
)
from brain.config import Settings
from contracts import Person

STRANGER = Person(id=str(uuid.uuid4()), name="Sam Stranger", short="Sam", initials="SS")


def base_settings(**kw) -> Settings:
    return Settings(
        _env_file=None,
        livekit_url=LIVEKIT_URL,
        livekit_api_key="test-key",
        livekit_api_secret="test-secret-that-is-long-enough-for-hs256",
        brain_internal_token=WORKER_TOKEN,
        **kw,
    )


def claims(sub: str = ALEX.id, **overrides) -> dict:
    now = int(time.time())
    body = {"iss": ISSUER, "aud": AUDIENCE, "sub": sub, "iat": now, "exp": now + 3600}
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not None}


def signed(secret: str = AUTH_SECRET, algorithm: str = "HS256", **kw) -> str:
    return jwt.encode(claims(**kw), secret, algorithm=algorithm)


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def settings() -> Settings:
    return base_settings(auth_secret=AUTH_SECRET)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# passwords


def test_a_password_hash_verifies_only_its_own_password():
    hashed = hash_password("correct horse battery staple")

    assert hashed.startswith("$argon2")
    assert "correct horse" not in hashed
    assert verify_password(hashed, "correct horse battery staple")
    assert not verify_password(hashed, "Correct horse battery staple")
    assert hash_password("correct horse battery staple") != hashed  # salted


@pytest.mark.parametrize("bad", ["", "not-a-hash", "$argon2id$v=19$m=1$broken"])
def test_a_malformed_hash_never_verifies(bad):
    assert verify_password(bad, "anything") is False


# issuing and verifying tokens


def test_an_issued_token_verifies_and_names_the_person(settings):
    before = datetime.now(UTC)
    token, expires_at = issue_token(ALEX.id, settings)

    found = TokenVerifier.from_settings(settings).verify(token)

    assert found["sub"] == ALEX.id
    assert (found["iss"], found["aud"]) == (ISSUER, AUDIENCE)
    assert jwt.get_unverified_header(token)["alg"] == "HS256"
    expected = before + timedelta(hours=settings.session_hours)
    assert abs((expires_at - expected).total_seconds()) < 5
    assert found["exp"] == int(expires_at.timestamp())


def test_sessions_last_session_hours():
    settings = base_settings(auth_secret=AUTH_SECRET, session_hours=2)

    _, expires_at = issue_token(ALEX.id, settings)

    assert abs((expires_at - datetime.now(UTC)) - timedelta(hours=2)) < timedelta(seconds=5)


def test_session_hours_default_to_12_and_are_at_least_1():
    assert base_settings().session_hours == 12
    with pytest.raises(ValidationError):
        base_settings(session_hours=0)


@pytest.mark.parametrize("secret", [None, "", "x" * 31])
def test_a_missing_or_short_secret_is_not_configured(secret):
    settings = base_settings(auth_secret=secret)

    assert not TokenVerifier.from_settings(settings).configured
    with pytest.raises(AuthNotConfigured):
        issue_token(ALEX.id, settings)
    with pytest.raises(AuthNotConfigured):
        TokenVerifier.from_settings(settings).verify(signed(secret="x" * 31))


@pytest.mark.parametrize(
    "token",
    [
        signed(secret="a-different-secret-that-is-also-long-enough-to-sign"),
        signed(exp=int(time.time()) - 120),
        signed(nbf=int(time.time()) + 120),
        signed(aud="authenticated"),
        signed(iss="https://abcd.example.co/auth/v1"),
        signed(aud=None),
        signed(iss=None),
        signed(exp=None),
        signed(sub=None),
        signed(sub=""),
        signed(algorithm="HS512"),
        signed(algorithm="HS384"),
        jwt.encode(claims(), key=None, algorithm="none"),
        "not-a-jwt",
    ],
    ids=[
        "wrong-secret",
        "expired",
        "not-yet-valid",
        "wrong-aud",
        "wrong-iss",
        "no-aud",
        "no-iss",
        "no-exp",
        "no-sub",
        "empty-sub",
        "hs512",
        "hs384",
        "alg-none",
        "junk",
    ],
)
def test_only_the_brains_own_hs256_tokens_are_accepted(settings, token):
    with pytest.raises(InvalidToken):
        TokenVerifier.from_settings(settings).verify(token)


def test_a_token_just_past_expiry_is_accepted_within_the_leeway(settings):
    token = signed(exp=int(time.time()) - 3)

    assert TokenVerifier.from_settings(settings).verify(token)["sub"] == ALEX.id


# on a real route


def test_an_issued_token_resolves_to_the_member_on_a_real_route(client, settings):
    token, _ = issue_token(ALEX.id, settings)

    response = client.post("/meetings", json={"title": "Standup"}, headers=bearer(token))

    assert response.status_code == 200, response.text
    assert response.json()["host_id"] == ALEX.id
    assert response.json()["team_id"] == TEAM.id
    assert client.get("/me", headers=bearer(token)).json()["id"] == ALEX.id


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
    [signed(exp=int(time.time()) - 120), signed(algorithm="HS512"), signed(aud="authenticated")],
    ids=["expired", "hs512", "wrong-aud"],
)
def test_an_invalid_token_is_401_on_a_real_route(client, token):
    response = client.get("/meetings", headers=bearer(token))

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_a_valid_token_for_someone_outside_the_workspace_is_403(client, settings):
    token, _ = issue_token(STRANGER.id, settings)

    response = client.get("/meetings", headers=bearer(token))

    assert response.status_code == 403
    assert response.json()["detail"] == "not a workspace member"


@pytest.mark.parametrize("secret", [None, "too-short"])
def test_no_auth_secret_is_503_never_an_anonymous_user(app, client, secret):
    from brain.api.deps import get_settings

    app.dependency_overrides[get_settings] = lambda: base_settings(auth_secret=secret)

    for headers in ({}, bearer(signed(secret="too-short"))):
        response = client.get("/meetings", headers=headers)
        assert response.status_code == 503
        assert "AUTH_SECRET" in response.json()["detail"]
