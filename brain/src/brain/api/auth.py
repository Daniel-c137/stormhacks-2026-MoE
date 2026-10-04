"""Sign-in checked by the brain (board -> brain): email and password, and creating an account.
Accounts come from `brain add-user`, an admin (POST /team/accounts), or an invited email signing
up (#128): a member of SIGNUP_TEAM_ID with no login yet. There is no open sign-up."""

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.concurrency import run_in_threadpool

from contracts import (
    AuthOptions,
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
from ..store import Conflict, Login, NotFound, Store
from .deps import current_user, get_login_limiter, get_settings, get_store

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
