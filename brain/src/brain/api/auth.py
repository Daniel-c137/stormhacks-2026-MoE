"""Email and password sign-in, checked by the brain (board -> brain). There is no public sign-up;
accounts come from `brain add-user`."""

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.concurrency import run_in_threadpool

from contracts import LoginRequest, LoginResponse, PasswordChange, Person

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
from ..store import Login, NotFound, Store
from .deps import current_user, get_login_limiter, get_settings, get_store

router = APIRouter(prefix="/auth", tags=["auth"])

MAX_EMAIL = 320
WRONG_LOGIN = "Wrong email or password"


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


@router.post("/password", status_code=204)
async def change_password(
    body: PasswordChange,
    user: Person = Depends(current_user),
    store: Store = Depends(get_store),
    limiter: LoginLimiter = Depends(get_login_limiter),
) -> Response:
    """Sessions already issued stay valid until they expire."""
    if len(body.new_password) < MIN_PASSWORD:
        raise HTTPException(
            status_code=422, detail=f"The new password must be at least {MIN_PASSWORD} characters"
        )
    if len(body.new_password) > MAX_PASSWORD:
        raise HTTPException(
            status_code=422, detail=f"The new password must be at most {MAX_PASSWORD} characters"
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
