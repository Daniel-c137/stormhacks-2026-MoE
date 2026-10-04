"""Supabase Auth access-token verification.

Supabase signs access tokens either with the project's legacy JWT secret (HS256) or with its
asymmetric signing keys (ES256 or RS256, `kid` in the header), whose public halves are served
at `<SUPABASE_URL>/auth/v1/.well-known/jwks.json`.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import jwt

from .config import Settings

LEEWAY = 10  # seconds of clock skew allowed on exp/nbf/iat
JWKS_TTL = 600.0  # Supabase's edge caches the JWKS for 10 minutes; don't hold keys longer
REFETCH_COOLDOWN = 30.0  # unknown kids force at most one refetch per this many seconds
ASYMMETRIC = ("ES256", "RS256")

Fetch = Callable[[str], Awaitable[dict[str, Any]]]


class AuthNotConfigured(Exception):
    """Neither SUPABASE_JWT_SECRET nor SUPABASE_URL is set."""


class InvalidToken(Exception):
    """The bearer token is not a valid access token for this project."""


class KeysUnavailable(Exception):
    """The project's JWKS could not be fetched and no keys are cached."""


async def fetch_json(url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


class Jwks:
    """The project's public signing keys, cached for `ttl` seconds.

    An unknown kid refetches once (keys may have rotated), unless the keys were just loaded.
    """

    def __init__(
        self,
        url: str,
        fetch: Fetch = fetch_json,
        ttl: float = JWKS_TTL,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.url = url
        self._fetch = fetch
        self._ttl = ttl
        self._clock = clock
        self._keys: dict[str, jwt.PyJWK] = {}
        self._loaded_at: float | None = None
        self._forced_at: float | None = None
        self._lock = asyncio.Lock()

    async def key(self, kid: str) -> jwt.PyJWK:
        loaded = False
        if self._stale():
            await self._refresh()
            loaded = True
        if kid not in self._keys and not loaded and self._may_force():
            self._forced_at = self._clock()
            await self._refresh()
        try:
            return self._keys[kid]
        except KeyError:
            raise InvalidToken("unknown signing key") from None

    def _stale(self) -> bool:
        return self._loaded_at is None or self._clock() - self._loaded_at >= self._ttl

    def _may_force(self) -> bool:
        return self._forced_at is None or self._clock() - self._forced_at >= REFETCH_COOLDOWN

    async def _refresh(self) -> None:
        async with self._lock:
            try:
                body = await self._fetch(self.url)
                keys = {}
                for entry in body.get("keys", []):
                    kid = entry.get("kid")
                    if not kid:
                        continue
                    try:
                        keys[kid] = jwt.PyJWK(entry)
                    except jwt.PyJWTError:
                        continue  # a key type this server can't use; skip it, keep the rest
            except (httpx.HTTPError, ValueError, AttributeError) as error:
                if not self._keys:
                    raise KeysUnavailable(f"could not fetch {self.url}") from error
                return  # keep serving the keys we have
            self._keys = keys
            self._loaded_at = self._clock()


class TokenVerifier:
    def __init__(
        self,
        *,
        secret: str | None,
        supabase_url: str | None,
        audience: str = "authenticated",
        jwks: Jwks | None = None,
    ):
        self.secret = secret or None
        base = supabase_url.rstrip("/") if supabase_url else None
        self.issuer = f"{base}/auth/v1" if base else None
        self.audience = audience
        self.jwks = jwks or (Jwks(f"{self.issuer}/.well-known/jwks.json") if base else None)

    @classmethod
    def from_settings(cls, settings: Settings) -> "TokenVerifier":
        return cls(
            secret=settings.supabase_jwt_secret,
            supabase_url=settings.supabase_url,
            audience=settings.supabase_jwt_audience,
        )

    @property
    def configured(self) -> bool:
        return bool(self.secret or self.jwks)

    async def verify(self, token: str) -> dict[str, Any]:
        """The token's claims. Raises AuthNotConfigured, InvalidToken or KeysUnavailable."""
        if not self.configured:
            raise AuthNotConfigured
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as error:
            raise InvalidToken("malformed token") from error
        alg = header.get("alg")
        if alg == "HS256" and self.secret:
            key: Any = self.secret
        elif alg in ASYMMETRIC and self.jwks:
            kid = header.get("kid")
            if not isinstance(kid, str) or not kid:
                raise InvalidToken("token has no key id")
            jwk = await self.jwks.key(kid)
            if jwk.algorithm_name != alg:
                raise InvalidToken("token algorithm does not match its key")
            key = jwk.key
        else:
            raise InvalidToken(f"unsupported token algorithm {alg!r}")
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[alg],
                audience=self.audience,
                issuer=self.issuer,
                leeway=LEEWAY,
                options={"require": ["exp", "sub", "aud"] + (["iss"] if self.issuer else [])},
            )
        except jwt.PyJWTError as error:
            raise InvalidToken(str(error)) from error
        if not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise InvalidToken("token has no subject")
        return claims
