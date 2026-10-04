"""Cancelling a subscription with its provider."""

from api.cancel.approvals import ApprovalError, verify_approval
from api.cancel.providers import provider_for
from api.models import Subscription


def cancel_subscription(subscription: Subscription, approval_token: str | None) -> None:
    """Cancels only with a signed approval token from the user's Approve tap (#49).

    Email text or assistant output can never stand in for the token.
    """
    if approval_token is None:
        raise ApprovalError("cancelling needs the user's Approve tap")
    verify_approval(approval_token, subscription_id=subscription.id)
    provider_for(subscription).cancel(subscription.external_id)
