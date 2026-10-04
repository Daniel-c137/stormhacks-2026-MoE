"""Google sign-in for the brain's own sessions (#128): OpenID Connect's authorization code flow
with PKCE, done server-side so no third-party auth service holds our users.

The flow's state, PKCE verifier, nonce and where to go next live in a short-lived cookie signed
with AUTH_SECRET, so only the browser that started a sign-in can finish it. Google's ID token is
verified against its published keys (signature, issuer, audience, expiry, nonce) and must carry a
verified email that Google is authoritative for (Gmail, or a Workspace account). The board gets a
one-time code, never the session token in a URL, and the code only works in the browser that
signed in: a second HttpOnly cookie carries its other half.
"""

import base64
import hashlib
import re
import secrets
import time
from dataclasses import dataclass, field
from urllib.parse import unquote, urlencode, urlsplit

import httpx
import jwt

from contracts import LoginResponse

from .auth import ALGORITHM

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
KEYS_URL = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS = ("https://accounts.google.com", "accounts.google.com")
GMAIL = ("gmail.com", "googlemail.com")
SCOPE = "openid email profile"

FLOW_COOKIE = "google_signin"
FLOW_COOKIE_PATH = "/"  # the brain may be served under a prefix such as /api
FLOW_SECONDS = 10 * 60  # time allowed at Google
FLOW_AUDIENCE = "google-signin"
HANDOFF_COOKIE = "google_handoff"  # binds the one-time code to the browser that signed in
CODE_SECONDS = 60  # the board swaps its one-time code at once
KEYS_SECONDS = 60 * 60  # Google rotates its keys rarely and publishes the new one early
UNSAFE = re.compile(r"[\\\x00-\x1f\x7f]")  # a backslash or an ASCII control character


class GoogleSignInFailed(Exception):
    """The callback can't be trusted or Google's answer doesn't check out. `reason` goes back to
    the sign-in page: state, cancelled, failed, unverified, not_invited or ambiguous (two teams
    invited the email)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason


@dataclass(frozen=True)
class Flow:
    state: str
    verifier: str
    nonce: str
    next: str

    @property
    def challenge(self) -> str:
        digest = hashlib.sha256(self.verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def safe_next(path: str | None) -> str:
    """A same-site path to land on after signing in; anything else is home. Browsers read a
    backslash as a slash and drop tabs and newlines, so `/\\evil.com` and `/<TAB>/evil.com` are
    //evil.com to them: a backslash or a control character, as given or once percent-decoded,
    is refused, and so is anything that isn't a single-slash path."""
    if not path or not path.startswith("/") or path.startswith("//"):
        return "/"
    if any(UNSAFE.search(text) for text in (path, unquote(path))):
        return "/"
    parts = urlsplit(path)
    return "/" if parts.scheme or parts.netloc else path


def new_flow(next_path: str | None) -> Flow:
    return Flow(
        state=secrets.token_urlsafe(32),
        verifier=secrets.token_urlsafe(64),
        nonce=secrets.token_urlsafe(32),
        next=safe_next(next_path),
    )


def authorize_url(client_id: str, redirect_uri: str, flow: Flow) -> str:
    query = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "state": flow.state,
        "nonce": flow.nonce,
        "code_challenge": flow.challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }
    return f"{AUTHORIZE_URL}?{urlencode(query)}"


def seal(flow: Flow, secret: str) -> str:
    now = int(time.time())
    claims = {
        "aud": FLOW_AUDIENCE,
        "iat": now,
        "exp": now + FLOW_SECONDS,
        "state": flow.state,
        "verifier": flow.verifier,
        "nonce": flow.nonce,
        "next": flow.next,
    }
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def unseal(cookie: str | None, state: str | None, secret: str) -> Flow:
    """The flow this browser started, if `state` is its state. Raises GoogleSignInFailed."""
    if not cookie or not state:
        raise GoogleSignInFailed("state", "no sign-in was started in this browser")
    try:
        claims = jwt.decode(cookie, secret, algorithms=[ALGORITHM], audience=FLOW_AUDIENCE)
    except jwt.PyJWTError as e:
        raise GoogleSignInFailed("state", f"the sign-in cookie is invalid: {e}") from None
    if not secrets.compare_digest(str(claims.get("state", "")), state):
        raise GoogleSignInFailed("state", "the state doesn't match this browser's sign-in")
    return Flow(claims["state"], claims["verifier"], claims["nonce"], safe_next(claims["next"]))


@dataclass
class GoogleKeys:
    """Google's published signing keys, fetched when a kid is new and at most hourly otherwise."""

    keys: dict[str, jwt.PyJWK] = field(default_factory=dict)
    fetched_at: float = 0.0

    async def key(self, kid: str, http: httpx.AsyncClient) -> jwt.PyJWK:
        stale = time.monotonic() - self.fetched_at > KEYS_SECONDS
        if kid not in self.keys or stale:
            response = await http.get(KEYS_URL)
            response.raise_for_status()
            self.keys = {k["kid"]: jwt.PyJWK(k) for k in response.json().get("keys", [])}
            self.fetched_at = time.monotonic()
        if kid not in self.keys:
            raise GoogleSignInFailed("failed", "the ID token is signed by an unpublished key")
        return self.keys[kid]


@dataclass
class GoogleIdentity:
    email: str
    name: str | None


async def verify_callback(
    code: str,
    flow: Flow,
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    keys: GoogleKeys,
    http: httpx.AsyncClient,
) -> GoogleIdentity:
    """Swaps the code (with the PKCE verifier) for an ID token and verifies it. Raises
    GoogleSignInFailed."""
    try:
        response = await http.post(
            TOKEN_URL,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": flow.verifier,
            },
        )
        response.raise_for_status()
        id_token = response.json()["id_token"]
        kid = jwt.get_unverified_header(id_token).get("kid", "")
        key = await keys.key(kid, http)
        claims = jwt.decode(
            id_token,
            key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=ISSUERS,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except GoogleSignInFailed:
        raise
    except (httpx.HTTPError, KeyError, ValueError, jwt.PyJWTError) as e:
        raise GoogleSignInFailed("failed", f"Google's answer doesn't check out: {e}") from None
    if not secrets.compare_digest(str(claims.get("nonce", "")), flow.nonce):
        raise GoogleSignInFailed("failed", "the ID token's nonce isn't this sign-in's")
    email = str(claims.get("email") or "").strip()
    if not email or claims.get("email_verified") is not True:
        raise GoogleSignInFailed("unverified", "Google hasn't verified this email")
    if not google_hosts(email, claims.get("hd")):
        raise GoogleSignInFailed("unverified", "Google isn't authoritative for this email")
    return GoogleIdentity(email=email, name=claims.get("name"))


def google_hosts(email: str, hd: object) -> bool:
    """Whether Google is authoritative for the email: a Gmail address, or a Google Workspace
    account (`hd`, its domain). Any other address can be put on a Google account, and Google
    keeps it verified after the mailbox changes hands, so it proves nothing about who owns it
    now: Google's guidance is to not link accounts by such an email."""
    domain = email.rpartition("@")[2].lower()
    return domain in GMAIL or (isinstance(hd, str) and bool(hd.strip()))


@dataclass
class OneTimeCodes:
    """Sessions waiting for the board to collect them, each by a code that works once and
    briefly, and only with the binding that went to the same browser in an HttpOnly cookie (so a
    code put in someone else's link signs nobody in). In process: a sign-in has to finish on the
    brain that started it."""

    pending: dict[str, tuple[LoginResponse, str, float]] = field(default_factory=dict)

    def issue(self, session: LoginResponse) -> tuple[str, str]:
        """The code for the URL and the binding for the browser's cookie."""
        now = time.monotonic()
        self.pending = {c: v for c, v in self.pending.items() if v[2] > now}
        code, binding = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        self.pending[code] = (session, binding, now + CODE_SECONDS)
        return code, binding

    def redeem(self, code: str, binding: str | None) -> LoginResponse | None:
        """The session, once: a wrong binding spends the code too."""
        found = self.pending.pop(code, None)
        if found is None or found[2] <= time.monotonic():
            return None
        session, expected, _ = found
        if not binding or not secrets.compare_digest(expected, binding):
            return None
        return session
