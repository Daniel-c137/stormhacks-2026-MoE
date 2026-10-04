"""The brain's own sign-in: argon2 password hashes, HS256 session tokens signed with AUTH_SECRET,
and a limit on failed logins per email."""

import math
import secrets
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import cache
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from .config import Settings

ALGORITHM = "HS256"
ISSUER = "brain"
AUDIENCE = "board"
LEEWAY = 10  # seconds of clock skew allowed on exp/nbf/iat
MIN_SECRET = 32  # a shorter AUTH_SECRET is treated as unset

MIN_PASSWORD = 10
MAX_PASSWORD = 1024  # argon2 hashes any length; this only bounds the work one request can ask for
MAX_LOGIN_FAILURES = 5
LOGIN_WINDOW_SECONDS = 15 * 60

_hasher = PasswordHasher()


class AuthNotConfigured(Exception):
    """AUTH_SECRET is unset or shorter than MIN_SECRET."""


class InvalidToken(Exception):
    """The bearer token is not a valid session token issued by this brain."""


NOT_CONFIGURED = f"AUTH_SECRET is not configured (at least {MIN_SECRET} characters)"


def signing_secret(settings: Settings) -> str | None:
    secret = settings.auth_secret
    return secret if secret and len(secret) >= MIN_SECRET else None


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(hashed: str, password: str) -> bool:
    try:
        return _hasher.verify(hashed, password)
    except (VerificationError, InvalidHashError):
        return False


@cache
def dummy_hash() -> str:
    """A hash no password matches, checked when the email is unknown so that a login takes as
    long whether or not the account exists."""
    return hash_password(secrets.token_urlsafe(32))


def issue_token(person_id: str, settings: Settings) -> tuple[str, datetime]:
    """A session token for the person and when it expires. Raises AuthNotConfigured."""
    secret = signing_secret(settings)
    if secret is None:
        raise AuthNotConfigured
    now = int(time.time())
    expires = now + int(timedelta(hours=settings.session_hours).total_seconds())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": person_id, "iat": now, "exp": expires}
    return jwt.encode(claims, secret, algorithm=ALGORITHM), datetime.fromtimestamp(expires, UTC)


class TokenVerifier:
    """Accepts only the brain's own HS256 tokens, with exp, sub, aud and iss."""

    def __init__(self, secret: str | None):
        self.secret = secret if secret and len(secret) >= MIN_SECRET else None

    @classmethod
    def from_settings(cls, settings: Settings) -> "TokenVerifier":
        return cls(signing_secret(settings))

    @property
    def configured(self) -> bool:
        return self.secret is not None

    def verify(self, token: str) -> dict[str, Any]:
        """The token's claims. Raises AuthNotConfigured or InvalidToken."""
        if self.secret is None:
            raise AuthNotConfigured
        try:
            claims = jwt.decode(
                token,
                self.secret,
                algorithms=[ALGORITHM],
                audience=AUDIENCE,
                issuer=ISSUER,
                leeway=LEEWAY,
                options={"require": ["exp", "sub", "aud", "iss"]},
            )
        except jwt.PyJWTError as error:
            raise InvalidToken(str(error)) from error
        if not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise InvalidToken("token has no subject")
        return claims


def login_key(email: str) -> str:
    return email.strip().lower()


class LoginLimiter:
    """Failed logins per email, in process memory: after MAX_LOGIN_FAILURES within the window,
    the email is refused until the oldest of them is a window old. A success clears the count."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        max_failures: int = MAX_LOGIN_FAILURES,
        window: float = LOGIN_WINDOW_SECONDS,
    ):
        self._clock = clock
        self._max = max_failures
        self._window = window
        self._failures: dict[str, deque[float]] = {}
        self._pruned_at = clock()

    def __len__(self) -> int:
        return len(self._failures)

    def retry_after(self, email: str) -> int | None:
        """Seconds until the email may try again, or None when it may now."""
        failures = self._recent(login_key(email))
        if failures is None or len(failures) < self._max:
            return None
        return max(1, math.ceil(failures[0] + self._window - self._clock()))

    def fail(self, email: str) -> None:
        self._prune()
        key = login_key(email)
        failures = self._recent(key)
        if failures is None:
            failures = self._failures[key] = deque()
        failures.append(self._clock())
        while len(failures) > self._max:
            failures.popleft()

    def clear(self, email: str) -> None:
        self._failures.pop(login_key(email), None)

    def _recent(self, key: str) -> deque[float] | None:
        failures = self._failures.get(key)
        if failures is None:
            return None
        since = self._clock() - self._window
        while failures and failures[0] <= since:
            failures.popleft()
        if not failures:
            del self._failures[key]
            return None
        return failures

    def _prune(self) -> None:
        """Forgets emails whose failures are all old, at most once a window."""
        now = self._clock()
        if now - self._pruned_at < self._window:
            return
        self._pruned_at = now
        for key in list(self._failures):
            self._recent(key)
