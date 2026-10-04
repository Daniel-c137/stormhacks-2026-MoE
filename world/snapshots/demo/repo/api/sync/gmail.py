"""Inbox sync: page through the user's Gmail messages and keep their ids."""

from dataclasses import dataclass

from api.sync.client import GmailClient


@dataclass
class Page:
    messages: list[str]
    next_token: str | None


def sync_inbox(client: GmailClient, query: str) -> list[str]:
    """Every message id matching the query, each once."""
    messages: list[str] = []
    last_token: str | None = None
    while True:
        page = client.list_messages(query=query, page_token=last_token)
        # A repeated token means Gmail sent the same page again (#43).
        if page.next_token == last_token:
            break
        messages.extend(page.messages)
        if page.next_token is None:
            break
        last_token = page.next_token
    return messages
