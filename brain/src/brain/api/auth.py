"""Sign-in checked by the brain (board -> brain): email and password, and creating an account.
Accounts come from `brain add-user`, an admin (POST /team/accounts), or an invited email signing
up (#128): a member of SIGNUP_TEAM_ID with no login yet. There is no open sign-up."""

import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse

from contracts import (
    AuthOptions,
    GoogleExchangeRequest,
    LoginRequest,
    LoginResponse,
    PasswordChange,
    Person,
    SignupRequest,
)

from ..accounts import InvalidAccount, clean_email, clean_name, invited_person, save_account
from ..auth import (
    MAX_PASSWORD,
    MIN_PASSWORD,
    NOT_CONFIGURED,
    LoginLimiter,
    dummy_hash,
    hash_password,
    issue_token,
    signing_secret,
    verify_password,
)
from ..config import Settings
from ..google_auth import (
    FLOW_COOKIE,
    FLOW_COOKIE_PATH,
    FLOW_SECONDS,
    GoogleKeys,
    GoogleSignInFailed,
    OneTimeCodes,
    authorize_url,
    new_flow,
    seal,
    unseal,
    verify_callback,
)
from ..store import Conflict, Login, NotFound, Store
from .deps import (
    current_user,
    get_google_keys,
    get_http_transport,
    get_login_limiter,
    get_one_time_codes,
    get_settings,
    get_store,
)

router = APIRouter(prefix="/auth", tags=["auth"])

MAX_EMAIL = 320
WRONG_LOGIN = "Wrong email or password"
NOT_INVITED = "This email hasn't been invited. Ask your team's admin to add you."
ALREADY_SIGNED_UP = "An account already uses this email. Sign in instead."
SIGNUP_NOT_CONFIGURED = "Sign-up isn't configured: set SIGNUP_TEAM_ID"


def google_configured(settings: Settings) -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def password_problem(password: str) -> str | None:
    if len(password) < MIN_PASSWORD:
        return f"The password must be at least {MIN_PASSWORD} characters"
    if len(password) > MAX_PASSWORD:
        return f"The password must be at most {MAX_PASSWORD} characters"
    return None


def too_many(retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail="Too many failed attempts; try again later",
        headers={"Retry-After": str(retry_after)},
    )


async def password_matches(login: Login | None, password: str) -> bool:
    """Checks a hash even when there is no login, so the time taken doesn't tell which accounts
    exist. Off the event loop: argon2 is deliberately slow."""
    hashed = login.password_hash if login else dummy_hash()
    ok = await run_in_threadpool(verify_password, hashed, password)
    return ok and login is not None


@router.post("/login")
async def login(
    body: LoginRequest,
    settings: Settings = Depends(get_settings),
    store: Store = Depends(get_store),
    limiter: LoginLimiter = Depends(get_login_limiter),
) -> LoginResponse:
    """A session token for the board to send as a bearer token. An unknown email and a wrong
    password get the same 401; after repeated failures the email waits (429)."""
    if signing_secret(settings) is None:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)
    email = body.email.strip()
    if (wait := limiter.retry_after(email)) is not None:
        raise too_many(wait)
    found: Login | None = None
    # an over-long password matches nothing, even if it starts with the right one
    if len(email) <= MAX_EMAIL and len(body.password) <= MAX_PASSWORD:
        try:
            found = await store.login_by_email(email)
        except NotFound:
            pass
    if not await password_matches(found, body.password[:MAX_PASSWORD]) or found is None:
        limiter.fail(email)
        raise HTTPException(status_code=401, detail=WRONG_LOGIN)
    try:
        person = await store.person(found.person_id)
    except NotFound:  # the login goes with the person, so only a race gets here
        limiter.fail(email)
        raise HTTPException(status_code=401, detail=WRONG_LOGIN) from None
    limiter.clear(email)
    token, expires_at = issue_token(person.id, settings)
    return LoginResponse(token=token, expires_at=expires_at, person=person)


@router.get("/options")
async def options(settings: Settings = Depends(get_settings)) -> AuthOptions:
    """What the sign-in page can offer, so it shows only what will work."""
    ready = signing_secret(settings) is not None
    return AuthOptions(
        signup=ready and bool(settings.signup_team_id),
        google=ready and google_configured(settings),
    )


@router.post("/signup", status_code=201)
async def sign_up(
    body: SignupRequest,
    settings: Settings = Depends(get_settings),
    store: Store = Depends(get_store),
    limiter: LoginLimiter = Depends(get_login_limiter),
) -> LoginResponse:
    """An invited email (a member of SIGNUP_TEAM_ID with no login yet) sets a password and is
    signed in. Anyone else gets 403, and repeated refusals lock the email (429) like failed
    logins; an email that already has an account gets 409. The mailbox isn't verified: the
    invitation is what admits the email (Google sign-in does verify it)."""
    if signing_secret(settings) is None:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)
    if not settings.signup_team_id:
        raise HTTPException(status_code=503, detail=SIGNUP_NOT_CONFIGURED)
    try:
        team = await store.team(settings.signup_team_id)
    except NotFound:
        raise HTTPException(status_code=503, detail=SIGNUP_NOT_CONFIGURED) from None
    try:
        name = clean_name(body.name)
        email = clean_email(body.email)
    except InvalidAccount as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    if problem := password_problem(body.password):
        raise HTTPException(status_code=422, detail=problem)
    if (wait := limiter.retry_after(email)) is not None:
        raise too_many(wait)
    try:
        await store.login_by_email(email)
        raise HTTPException(status_code=409, detail=ALREADY_SIGNED_UP)
    except NotFound:
        pass
    invited = await invited_person(store, team, email)
    if invited is None:
        limiter.fail(email)
        raise HTTPException(status_code=403, detail=NOT_INVITED)
    hashed = await run_in_threadpool(hash_password, body.password)
    try:
        person = await save_account(
            store, team, name=name, email=email, password_hash=hashed, person=invited
        )
    except Conflict:  # only a race with another sign-up for the same email gets here
        raise HTTPException(status_code=409, detail=ALREADY_SIGNED_UP) from None
    limiter.clear(email)
    token, expires_at = issue_token(person.id, settings)
    return LoginResponse(token=token, expires_at=expires_at, person=person)


# Google (#128)


@router.get("/google")
async def google_sign_in(
    request: Request,
    next: str | None = None,
    settings: Settings = Depends(get_settings),
) -> RedirectResponse:
    """Sends the browser to Google, with this sign-in's state, nonce and PKCE verifier sealed in
    a short-lived cookie that only this browser carries back."""
    secret = signing_secret(settings)
    if secret is None or not google_configured(settings):
        raise HTTPException(status_code=503, detail="Google sign-in isn't configured")
    flow = new_flow(next)
    redirect_uri = callback_url(request, settings)
    response = RedirectResponse(
        authorize_url(settings.google_client_id or "", redirect_uri, flow), status_code=302
    )
    response.set_cookie(
        FLOW_COOKIE,
        seal(flow, secret),
        max_age=FLOW_SECONDS,
        path=FLOW_COOKIE_PATH,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.get("/google/callback", name="google_callback")
async def google_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    google_signin: str | None = Cookie(default=None),
    settings: Settings = Depends(get_settings),
    store: Store = Depends(get_store),
    keys: GoogleKeys = Depends(get_google_keys),
    codes: OneTimeCodes = Depends(get_one_time_codes),
    transport: httpx.AsyncBaseTransport | None = Depends(get_http_transport),
) -> RedirectResponse:
    """Where Google sends the browser back. Signs in the account with Google's verified email,
    or an invited email (whose login is then reserved for Google), and sends the browser to the
    board's /login with a one-time code (`google`) or why it failed (`google_error`)."""

    def back(**query: str) -> RedirectResponse:
        board = (settings.board_url or "").rstrip("/")  # "": this site's /login
        response = RedirectResponse(f"{board}/login?{urlencode(query)}", status_code=302)
        response.delete_cookie(FLOW_COOKIE, path=FLOW_COOKIE_PATH)
        return response

    secret = signing_secret(settings)
    if secret is None or not google_configured(settings):
        return back(google_error="failed")
    try:
        flow = unseal(google_signin, state, secret)
        if error:
            raise GoogleSignInFailed("cancelled", error)
        if not code:
            raise GoogleSignInFailed("failed", "Google sent no code")
        async with httpx.AsyncClient(transport=transport, timeout=10) as http:
            identity = await verify_callback(
                code,
                flow,
                client_id=settings.google_client_id or "",
                client_secret=settings.google_client_secret or "",
                redirect_uri=callback_url(request, settings),
                keys=keys,
                http=http,
            )
        person = await google_person(store, settings, identity.email)
        if person is None:
            raise GoogleSignInFailed("not_invited", identity.email)
    except GoogleSignInFailed as e:
        return back(google_error=e.reason)
    token, expires_at = issue_token(person.id, settings)
    session = LoginResponse(token=token, expires_at=expires_at, person=person)
    return back(google=codes.issue(session), next=flow.next)


def callback_url(request: Request, settings: Settings) -> str:
    """The redirect URI registered with Google: GOOGLE_REDIRECT_URL behind a proxy, else this
    brain's own callback as the request reached it."""
    return settings.google_redirect_url or str(request.url_for("google_callback"))


async def google_person(store: Store, settings: Settings, email: str) -> Person | None:
    """Whoever signs in with this email, else the sign-up team's invited member with it, whose
    login is then reserved with a password nobody knows: the account is theirs through Google,
    and nobody can sign up for that email with a password. None for anyone else."""
    try:
        return await store.person((await store.login_by_email(email)).person_id)
    except NotFound:
        pass
    if not settings.signup_team_id:
        return None
    try:
        team = await store.team(settings.signup_team_id)
    except NotFound:
        return None
    invited = await invited_person(store, team, email)
    if invited is None:
        return None
    unusable = await run_in_threadpool(hash_password, secrets.token_urlsafe(32))
    try:
        await store.set_login(invited.id, invited.email or email, unusable)
    except Conflict:  # someone signed up for the email meanwhile
        return None
    return invited


@router.post("/google/exchange")
async def google_exchange(
    body: GoogleExchangeRequest, codes: OneTimeCodes = Depends(get_one_time_codes)
) -> LoginResponse:
    """The board swaps the one-time code from the callback for the session. Works once, within
    a minute."""
    session = codes.redeem(body.code)
    if session is None:
        raise HTTPException(status_code=401, detail="This sign-in has expired. Try again.")
    return session


@router.post("/password", status_code=204)
async def change_password(
    body: PasswordChange,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    limiter: LoginLimiter = Depends(get_login_limiter),
) -> Response:
    """Sessions already issued stay valid until they expire."""
    if problem := password_problem(body.new_password):
        raise HTTPException(
            status_code=422, detail=problem.replace("The password", "The new password")
        )
    try:
        saved: Login | None = await store.login(user.id)
    except NotFound:
        saved = None
    email = saved.email if saved else user.id
    if (wait := limiter.retry_after(email)) is not None:
        raise too_many(wait)
    # an over-long password matches nothing, even if it starts with the right one
    candidate = saved if len(body.current_password) <= MAX_PASSWORD else None
    if not await password_matches(candidate, body.current_password[:MAX_PASSWORD]) or not saved:
        limiter.fail(email)
        raise HTTPException(status_code=401, detail="The current password is wrong")
    hashed = await run_in_threadpool(hash_password, body.new_password)
    await store.set_login(user.id, saved.email, hashed)
    limiter.clear(email)
    return Response(status_code=204)
