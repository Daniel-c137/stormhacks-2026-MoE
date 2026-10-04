import pytest

from brain.sealing import Unsealable, seal, unseal

SECRET = "a-session-signing-secret-of-at-least-32-characters"


def test_a_sealed_secret_is_not_readable_and_opens_with_the_same_key():
    sealed = seal("jira-api-token", SECRET)

    assert "jira-api-token" not in sealed
    assert unseal(sealed, SECRET) == "jira-api-token"


def test_sealing_twice_gives_different_text():
    assert seal("jira-api-token", SECRET) != seal("jira-api-token", SECRET)


def test_another_key_or_a_changed_text_does_not_open_it():
    sealed = seal("jira-api-token", SECRET)

    with pytest.raises(Unsealable):
        unseal(sealed, SECRET + "-changed")
    with pytest.raises(Unsealable):
        unseal(sealed[:-4] + "AAAA", SECRET)
    with pytest.raises(Unsealable):
        unseal("not sealed at all", SECRET)
