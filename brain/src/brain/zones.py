"""Dates as the team sees them.

Timestamps are stored and exchanged in UTC. A date that reaches a person or a prompt (a meeting's
day, "today", a due date) is read in the team's time zone, TeamSettings.timezone.
"""

import logging
from datetime import UTC, date, datetime, tzinfo
from functools import cache
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, available_timezones

if TYPE_CHECKING:
    from brain.store import Store

log = logging.getLogger(__name__)


@cache
def known_zones() -> frozenset[str]:
    """IANA names exactly as the tz database spells them. Checked against the list rather than
    by loading, so a case-insensitive file system cannot let "america/vancouver" through."""
    return frozenset(available_timezones()) | {"UTC"}


def is_zone(name: str) -> bool:
    return name in known_zones()


def team_zone(name: str | None) -> tzinfo:
    """The zone for a saved name. PUT /settings accepts only known names; anything else (a zone
    this machine lacks) falls back to UTC rather than failing a request."""
    if not name or name == "UTC":
        return UTC
    if not is_zone(name):
        log.warning("Unknown time zone %r; using UTC", name[:60])
        return UTC
    return ZoneInfo(name)


async def zone_of(store: "Store", team_id: str) -> tzinfo:
    return team_zone((await store.settings(team_id)).timezone)


def now() -> datetime:
    """The current time, in UTC. Tests replace it."""
    return datetime.now(UTC)


def today(zone: tzinfo) -> date:
    """The team's date right now."""
    return now().astimezone(zone).date()


def local_date(when: datetime | None, zone: tzinfo) -> date | None:
    """The day `when` falls on in `zone`. A time without a zone is taken as UTC."""
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(zone).date()
