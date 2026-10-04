"""Sign-in through the brain: POST /auth/login and POST /auth/password, with real sessions (no
auth override). There is no public sign-up; accounts come from `brain add-user`."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from api_support import ALEX, AUTH_SECRET, SARAH
from fastapi.testclient import TestClient
from test_auth import base_settings, bearer

from brain import auth
from brain.api.deps import get_login_limiter, get_settings
from brain.auth import LoginLimiter, hash_password, verify_password
from contracts import LoginResponse

ALEX_EMAIL = "alex@example.com"
ALEX_PASSWORD = "alex-password-1"
SARAH_EMAIL = "Sarah.Kim@Example.com"
SARAH_PASSWORD = "sarah-password-1"
HASHES = {
    ALEX_PASSWORD: hash_password(ALEX_PASSWORD),
    SARAH_PASSWORD: hash_password(SARAH_PASSWORD),
}
WRONG = "Wrong email or password"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def settings():
    return base_settings(auth_secret=AUTH_SECRET, session_hours=12)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def limiter(app, clock) -> LoginLimiter:
    limiter = LoginLimiter(clock=clock)
    app.dependency_overrides[get_login_limiter] = lambda: limiter
    return limiter


@pytest.fixture
def client(app, store, limiter) -> TestClient:
    async def seed():
        await store.set_login(ALEX.id, ALEX_EMAIL, HASHES[ALEX_PASSWORD])
        await store.set_login(SARAH.id, SARAH_EMAIL, HASHES[SARAH_PASSWORD])

    asyncio.run(seed())
    return TestClient(app)


def log_in(client: TestClient, email: str, password: str):
    return client.post("/auth/login", json={"email": email, "password": password})


# logging in


def test_a_member_logs_in_and_the_token_is_their_session(client):
    before = datetime.now(UTC)
    response = log_in(client, ALEX_EMAIL, ALEX_PASSWORD)

    assert response.status_code == 200, response.text
    body = LoginResponse.model_validate(response.json())
    assert body.person == ALEX
    expected = before + timedelta(hours=12)
    assert abs((body.expires_at - expected).total_seconds()) < 5
    me = client.get("/me", headers=bearer(body.token))
    assert me.status_code == 200, me.text
    assert me.json()["id"] == ALEX.id
    listed = client.get("/meetings", headers=bearer(body.token))
    assert listed.status_code == 200


def test_the_response_never_carries_the_password_or_its_hash(client):
    text = log_in(client, ALEX_EMAIL, ALEX_PASSWORD).text

    assert ALEX_PASSWORD not in text
    assert "argon2" not in text
    assert "password" not in text


def test_the_email_is_matched_ignoring_case_and_surrounding_space(client):
    for email in ("ALEX@Example.COM", "  alex@example.com ", "sarah.kim@example.com"):
        password = SARAH_PASSWORD if "sarah" in email else ALEX_PASSWORD
        response = log_in(client, email, password)
        assert response.status_code == 200, (email, response.text)


def test_an_unknown_email_and_a_wrong_password_get_the_same_401(client):
    unknown = log_in(client, "nobody@example.com", ALEX_PASSWORD)
    wrong = log_in(client, ALEX_EMAIL, "not-the-password")

    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json() == {"detail": WRONG}


def test_an_unknown_email_still_checks_a_password_hash(client, monkeypatch):
    """So the time taken doesn't tell which accounts exist."""
    checked: list[str] = []

    def spy(hashed: str, password: str) -> bool:
        checked.append(hashed)
        return verify_password(hashed, password)

    monkeypatch.setattr("brain.api.auth.verify_password", spy)

    assert log_in(client, "nobody@example.com", "whatever-password").status_code == 401
    assert len(checked) == 1
    assert checked[0].startswith("$argon2")


def test_five_failures_lock_the_email_for_the_window_with_retry_after(client, clock):
    for _ in range(5):
        assert log_in(client, ALEX_EMAIL, "not-the-password").status_code == 401

    clock.now += 60
    locked = log_in(client, ALEX_EMAIL, ALEX_PASSWORD)

    assert locked.status_code == 429
    retry_after = int(locked.headers["retry-after"])
    assert 0 < retry_after <= 15 * 60 - 60
    # another email is unaffected
    assert log_in(client, SARAH_EMAIL, SARAH_PASSWORD).status_code == 200

    clock.now += retry_after
    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 200


def test_failures_count_per_email_ignoring_case(client):
    for email in ("alex@example.com", "ALEX@example.com", "Alex@Example.com", " alex@example.com"):
        assert log_in(client, email, "not-the-password").status_code == 401
    assert log_in(client, "ALEX@EXAMPLE.COM", "not-the-password").status_code == 401

    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 429


def test_unknown_emails_are_limited_too(client):
    for _ in range(5):
        assert log_in(client, "nobody@example.com", "guess-password").status_code == 401

    assert log_in(client, "nobody@example.com", "guess-password").status_code == 429


def test_a_successful_login_clears_the_failures(client):
    for _ in range(4):
        assert log_in(client, ALEX_EMAIL, "not-the-password").status_code == 401
    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 200

    for _ in range(4):
        assert log_in(client, ALEX_EMAIL, "not-the-password").status_code == 401
    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 200


def test_old_failures_fall_out_of_the_window(client, clock):
    for _ in range(4):
        assert log_in(client, ALEX_EMAIL, "not-the-password").status_code == 401
    clock.now += 15 * 60 + 1

    for _ in range(4):
        assert log_in(client, ALEX_EMAIL, "not-the-password").status_code == 401
    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 200


@pytest.mark.parametrize("secret", [None, "too-short"])
def test_login_without_an_auth_secret_is_503(app, client, secret):
    app.dependency_overrides[get_settings] = lambda: base_settings(auth_secret=secret)

    response = log_in(client, ALEX_EMAIL, ALEX_PASSWORD)

    assert response.status_code == 503
    assert "AUTH_SECRET" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [{"email": ALEX_EMAIL}, {"password": ALEX_PASSWORD}, {"email": ALEX_EMAIL, "password": 5}],
)
def test_a_malformed_login_is_422(client, body):
    assert client.post("/auth/login", json=body).status_code == 422


def test_there_is_no_public_sign_up(client):
    for path in ("/auth/signup", "/auth/register", "/auth/users"):
        response = client.post(path, json={"email": "new@example.com", "password": "x" * 12})
        assert response.status_code in (404, 405), path


# changing the password


def session(client: TestClient, email: str = ALEX_EMAIL, password: str = ALEX_PASSWORD) -> dict:
    response = log_in(client, email, password)
    assert response.status_code == 200, response.text
    return bearer(response.json()["token"])


def change(client: TestClient, headers: dict, current: str, new: str):
    return client.post(
        "/auth/password",
        json={"current_password": current, "new_password": new},
        headers=headers,
    )


def test_a_member_changes_their_password(client):
    headers = session(client)

    response = change(client, headers, ALEX_PASSWORD, "a-brand-new-password")

    assert response.status_code == 204, response.text
    assert response.content == b""
    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 401
    assert log_in(client, ALEX_EMAIL, "a-brand-new-password").status_code == 200
    # someone else's password is untouched
    assert log_in(client, SARAH_EMAIL, SARAH_PASSWORD).status_code == 200


def test_the_new_password_is_stored_hashed(client, store):
    change(client, session(client), ALEX_PASSWORD, "a-brand-new-password")

    saved = asyncio.run(store.login(ALEX.id))
    assert saved.password_hash.startswith("$argon2")
    assert verify_password(saved.password_hash, "a-brand-new-password")
    assert saved.email == ALEX_EMAIL


def test_a_wrong_current_password_is_401_and_counts_as_a_failure(client):
    headers = session(client)

    for _ in range(5):
        response = change(client, headers, "not-the-password", "a-brand-new-password")
        assert response.status_code == 401
    assert change(client, headers, ALEX_PASSWORD, "a-brand-new-password").status_code == 429
    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 429


def test_a_new_password_under_10_characters_is_422(client):
    headers = session(client)

    response = change(client, headers, ALEX_PASSWORD, "short-pw1")

    assert response.status_code == 422
    assert "10" in response.json()["detail"]
    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 200


def test_changing_the_password_needs_a_session(client):
    response = change(client, {}, ALEX_PASSWORD, "a-brand-new-password")

    assert response.status_code == 401
    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 200


def test_the_limiter_is_shared_by_the_app_between_requests(app, store):
    """Without an override, the app keeps one limiter for its life."""
    asyncio.run(store.set_login(ALEX.id, ALEX_EMAIL, HASHES[ALEX_PASSWORD]))
    client = TestClient(app)

    for _ in range(5):
        assert log_in(client, ALEX_EMAIL, "not-the-password").status_code == 401

    assert log_in(client, ALEX_EMAIL, ALEX_PASSWORD).status_code == 429


def test_the_limiter_forgets_emails_whose_failures_are_old():
    clock = Clock()
    limiter = LoginLimiter(clock=clock)
    for n in range(50):
        limiter.fail(f"user{n}@example.com")
    clock.now += auth.LOGIN_WINDOW_SECONDS + 1

    limiter.fail("fresh@example.com")

    assert len(limiter) == 1
