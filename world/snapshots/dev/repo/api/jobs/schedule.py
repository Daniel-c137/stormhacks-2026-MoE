"""Recurring jobs. Times are UTC."""

from api.jobs.renewals import renewal_warnings
from api.jobs.runner import cron
from api.jobs.weekly_email import weekly_savings_email


def register() -> None:
    cron("0 0 * * *", renewal_warnings)
    cron("0 14 * * MON", weekly_savings_email)
