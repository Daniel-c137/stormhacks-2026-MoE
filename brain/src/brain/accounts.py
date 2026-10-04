"""Accounts: a person on a team with an email and password login, and who is an admin. Shared by
`brain add-user`, `brain set-admin` and POST /team/accounts; there is no public sign-up."""

import re
import secrets
import uuid

from contracts import Person, Team

from .store import NotFound, Store

MAX_NAME_LENGTH = 80
MAX_EMAIL = 320
EMAIL = re.compile(r"[^@\s]+@[^@\s]+")


class InvalidAccount(ValueError):
    """A blank or over-long name, or something that is not an email address."""


class LastAdmin(Exception):
    """Revoking would leave the team without an admin."""


def clean_name(name: str) -> str:
    """Whitespace collapsed; raises InvalidAccount for a blank or over-long name."""
    name = " ".join(name.split())
    if not name:
        raise InvalidAccount("Name is required")
    if len(name) > MAX_NAME_LENGTH:
        raise InvalidAccount(f"Name must be at most {MAX_NAME_LENGTH} characters")
    return name


def clean_email(email: str) -> str:
    """Trimmed, case kept; raises InvalidAccount unless it looks like an email address."""
    email = email.strip()
    if len(email) > MAX_EMAIL or not EMAIL.fullmatch(email):
        raise InvalidAccount(f"{email!r} is not an email address")
    return email


def generate_password() -> str:
    """24 URL-safe characters (144 random bits), shown once to whoever made the account."""
    return secrets.token_urlsafe(18)


def name_fields(name: str) -> dict[str, str]:
    """The short name and initials shown on tiles, from a cleaned full name."""
    words = name.split()
    initials = (words[0][0] + (words[-1][0] if len(words) > 1 else "")).upper()
    return {"name": name, "short": words[0], "initials": initials}


async def existing_person(store: Store, team: Team, email: str) -> Person | None:
    """Whoever already signs in with the email, else the team's member with that email."""
    try:
        return await store.person((await store.login_by_email(email)).person_id)
    except NotFound:
        pass
    for member in await store.members(team.id):
        if member.email and member.email.lower() == email.lower():
            return member
    return None


async def save_account(
    store: Store,
    team: Team,
    *,
    name: str,
    email: str,
    password_hash: str,
    person: Person | None = None,
    title: str | None = None,
    is_admin: bool | None = None,
) -> Person:
    """Puts `person` (or a new one) on the team with the name and email, then sets their login.
    `title` and `is_admin` change only when given. Raises Conflict when another person already
    signs in with the email; the person is saved by then."""
    person = person or Person(id=str(uuid.uuid4()), **name_fields(name))
    changes: dict[str, object] = name_fields(name) | {"email": email}
    if title is not None:
        changes["title"] = title
    if is_admin is not None:
        changes["is_admin"] = is_admin
    person = await store.upsert_person(person.model_copy(update=changes), team.id)
    await store.set_login(person.id, email, password_hash)
    return person


async def set_admin(store: Store, person_id: str, is_admin: bool) -> Person:
    """Grants or revokes admin. Raises LastAdmin rather than leave the person's team with none,
    and NotFound for a missing person."""
    person = await store.person(person_id)
    if person.is_admin and not is_admin:
        try:
            team = await store.team_for_user(person.id)
        except NotFound:
            team = None
        if team and not any(m.is_admin for m in await store.members(team.id) if m.id != person.id):
            raise LastAdmin(f"{person.name} is the last admin of team {team.id}")
    if person.is_admin == is_admin:
        return person
    return await store.update_person(person.model_copy(update={"is_admin": is_admin}))
