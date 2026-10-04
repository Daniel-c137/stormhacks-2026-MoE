"""Accounts: a person on a team with an email and password login, and who is an admin. Shared by
`brain add-user`, `brain set-admin`, POST /team/accounts and invite-only sign-up (#128). An
invited person is on a team with an email and no login yet, until they sign up (#143)."""

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


class InvitedTwice(Exception):
    """More than one team invited the email, so whose account it is isn't clear."""


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


def name_from_email(email: str) -> str:
    """A stand-in name for someone invited by email only, from the address's local part
    ("priya.natarajan@…" is "Priya Natarajan"), until they choose their own."""
    words = re.split(r"[._+\-]+", email.partition("@")[0])
    return " ".join(w[:1].upper() + w[1:] for w in words if w)[:MAX_NAME_LENGTH] or email


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


async def invited_person(store: Store, email: str) -> tuple[Person, Team] | None:
    """The person an admin invited with this email (ignoring case), and the team they join: on
    a team with no login yet. None for anyone else, including someone who already signs in.
    Raises InvitedTwice when several teams invited the email."""
    found = await store.invited_people(email)
    if not found:
        return None
    if len(found) > 1:
        raise InvitedTwice(email)
    try:
        return found[0], await store.team_for_user(found[0].id)
    except NotFound:  # only a race with the team's removal gets here
        return None


async def invite_person(
    store: Store,
    team: Team,
    *,
    name: str,
    email: str,
    title: str | None = None,
    is_admin: bool = False,
) -> Person:
    """A new person on the team with the name and email and no login: they create it themselves
    with sign-up or Google."""
    person = Person(
        id=str(uuid.uuid4()), **name_fields(name), email=email, title=title, is_admin=is_admin
    )
    return await store.upsert_person(person, team.id)


async def accept_invite(
    store: Store, person: Person, password_hash: str, *, name: str | None = None
) -> Person:
    """Gives the invited person their first login, with the email they were invited with, and
    the name they chose (if given). Raises Conflict, changing nothing, when they or the email
    have a login already: another sign-up or Google got there first."""
    assert person.email, "an invited person has an email"
    await store.add_login(person.id, person.email, password_hash)
    if name is None:
        return await store.person(person.id)
    current = await store.person(person.id)
    return await store.update_person(current.model_copy(update=name_fields(name)))


async def signs_in(store: Store, person_id: str) -> bool:
    """Whether the person has a login (an invited person has none until they sign up)."""
    try:
        await store.login(person_id)
    except NotFound:
        return False
    return True


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
        if team is not None:
            # an invited admin can't sign in yet, so they don't count
            others = [m for m in await store.members(team.id) if m.is_admin and m.id != person.id]
            if not [m for m in others if await signs_in(store, m.id)]:
                raise LastAdmin(f"{person.name} is the last admin of team {team.id}")
    if person.is_admin == is_admin:
        return person
    return await store.update_person(person.model_copy(update={"is_admin": is_admin}))
