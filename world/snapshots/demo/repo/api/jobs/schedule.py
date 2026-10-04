"""Recurring jobs. Times are UTC."""

from api.jobs.renewals import renewal_warnings
from api.jobs.runner import cron
from api.jobs.weekly_email import weekly_savings_email


def register() -> None:
    # Once a day; a renewal found after this run is warned the next day (#36).
    cron("0 6 * * *", renewal_warnings)
    cron("0 14 * * MON", weekly_savings_email)
