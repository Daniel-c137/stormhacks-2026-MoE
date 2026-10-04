"""Accounts an admin creates from the settings page (board -> brain): with a password the server
makes and returns once, or as an invite with no login, which the person then creates on the
sign-in page (#143). Nothing is emailed."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from contracts import CreateAccountRequest, CreateAccountResponse, Person

from ..accounts import (
    MAX_NAME_LENGTH,
    InvalidAccount,
    clean_email,
    clean_name,
    existing_person,
    generate_password,
    invite_person,
    save_account,
)
from ..auth import hash_password
from ..store import Conflict, Store
from .deps import get_store, require_admin, user_team

router = APIRouter(tags=["team"])

EMAIL_IN_USE = "This email is already on the team or has an account"
INVITED_ELSEWHERE = "Another team has already invited this email"


@router.post("/team/accounts", status_code=201)
async def create_account(
    body: CreateAccountRequest,
    admin: Person = Depends(require_admin),
    store: Store = Depends(get_store),
) -> CreateAccountResponse:
    """A person on the admin's own team with an email and password login. The password is
    generated here and returned only in this response; the person can change it afterwards.
    With invite, the person gets no login and the password is None: they sign up themselves."""
    team = await user_team(store, admin)
    try:
        name = clean_name(body.name)
        email = clean_email(body.email)
    except InvalidAccount as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    title = " ".join((body.title or "").split()) or None
    if title and len(title) > MAX_NAME_LENGTH:
        raise HTTPException(
            status_code=422, detail=f"Title must be at most {MAX_NAME_LENGTH} characters"
        )
    if await existing_person(store, team, email) is not None:
        raise HTTPException(status_code=409, detail=EMAIL_IN_USE)
    # A second team's invite would leave the email unable to sign up (which team?), and a login
    # here would strand the first team's invite; nobody can remove a member yet.
    if await store.invited_people(email):
        raise HTTPException(status_code=409, detail=INVITED_ELSEWHERE)
    if body.invite:
        person = await invite_person(
            store, team, name=name, email=email, title=title, is_admin=body.is_admin
        )
        return CreateAccountResponse(person=person, password=None)
    password = generate_password()
    hashed = await run_in_threadpool(hash_password, password)  # argon2 is deliberately slow
    try:
        person = await save_account(
            store,
            team,
            name=name,
            email=email,
            password_hash=hashed,
            title=title,
            is_admin=body.is_admin,
        )
    except Conflict:  # only a race with another account for the same email gets here
        raise HTTPException(status_code=409, detail=EMAIL_IN_USE) from None
    return CreateAccountResponse(person=person, password=password)
