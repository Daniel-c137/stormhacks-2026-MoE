"""Secrets a team hands the brain (its Jira API token), encrypted before they are stored.

The key comes from AUTH_SECRET, so a database dump alone does not reveal them. Changing
AUTH_SECRET makes what was sealed unreadable; it then has to be entered again."""

import base64

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class Unsealable(ValueError):
    """Sealed with another key, or not sealed text at all."""


def _fernet(secret: str) -> Fernet:
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"brain sealed secrets")
    return Fernet(base64.urlsafe_b64encode(key.derive(secret.encode())))


def seal(text: str, secret: str) -> str:
    return _fernet(secret).encrypt(text.encode()).decode()


def unseal(sealed: str, secret: str) -> str:
    try:
        return _fernet(secret).decrypt(sealed.encode()).decode()
    except (InvalidToken, ValueError) as e:
        raise Unsealable("sealed with another key") from e
