"""Profile, team and workspace settings over HTTP."""

import pytest
from api_support import ADMIN, ALEX, OUTSIDER, SARAH, TEAM

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
MAX_PHOTO_BYTES = 2 * 1024 * 1024


def upload(client, data: bytes, filename: str = "me.png", content_type: str = "image/png"):
    return client.post("/me/photo", files={"file": (filename, data, content_type)})


# profile


def test_me_is_the_callers_own_profile(client_as):
    response = client_as(ALEX).get("/me")

    assert response.status_code == 200
    assert response.json()["id"] == ALEX.id
    assert response.json()["name"] == ALEX.name


def test_renaming_trims_and_recomputes_short_name_and_initials(client_as):
    alex = client_as(ALEX)

    response = alex.patch("/me", json={"name": "  Alexandra   de la Cruz  "})

    assert response.status_code == 200
    me = response.json()
    assert me["name"] == "Alexandra de la Cruz"
    assert me["short"] == "Alexandra"
    assert me["initials"] == "AC"
    assert alex.get("/me").json() == me
    member = next(p for p in client_as(SARAH).get("/team/members").json() if p["id"] == ALEX.id)
    assert member["name"] == "Alexandra de la Cruz"


def test_a_one_word_name_has_one_initial(client_as):
    me = client_as(ALEX).patch("/me", json={"name": "prince"}).json()

    assert (me["short"], me["initials"]) == ("prince", "P")


def test_a_blank_or_overlong_name_is_rejected(client_as):
    alex = client_as(ALEX)

    assert alex.patch("/me", json={"name": "   "}).status_code == 422
    assert alex.patch("/me", json={"name": "x" * 81}).status_code == 422
    assert alex.get("/me").json()["name"] == ALEX.name


def test_an_empty_profile_update_changes_nothing(client_as):
    response = client_as(ALEX).patch("/me", json={})

    assert response.status_code == 200
    assert response.json()["name"] == ALEX.name


# photo


def test_an_uploaded_photo_is_served_to_teammates(client_as):
    me = upload(client_as(ALEX), PNG).json()

    url = me["photo_url"]
    assert url.split("?")[0] == f"/people/{ALEX.id}/photo"
    response = client_as(SARAH).get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == PNG


def test_the_type_comes_from_the_bytes_not_the_file_name(client_as):
    alex = client_as(ALEX)

    jpeg = upload(alex, JPEG, filename="photo.png", content_type="image/png")
    assert jpeg.status_code == 200
    assert alex.get(jpeg.json()["photo_url"]).headers["content-type"] == "image/jpeg"

    gif = upload(alex, b"GIF89a" + b"\x00" * 64, filename="photo.png")
    assert gif.status_code == 415
    script = upload(alex, b"<script>alert(1)</script>", filename="photo.jpg")
    assert script.status_code == 415


def test_a_photo_over_two_megabytes_is_rejected(client_as):
    alex = client_as(ALEX)

    assert upload(alex, PNG + b"\x00" * (MAX_PHOTO_BYTES - len(PNG))).status_code == 200
    too_big = upload(alex, PNG + b"\x00" * (MAX_PHOTO_BYTES - len(PNG) + 1))
    assert too_big.status_code == 413


def test_an_empty_upload_is_rejected(client_as):
    assert upload(client_as(ALEX), b"").status_code == 422


def test_a_deleted_photo_is_no_longer_served(client_as):
    alex = client_as(ALEX)
    upload(alex, PNG)

    response = alex.delete("/me/photo")

    assert response.status_code == 200
    assert response.json()["photo_url"] is None
    assert alex.get(f"/people/{ALEX.id}/photo").status_code == 404


def test_another_teams_photos_look_like_they_do_not_exist(client_as):
    upload(client_as(ALEX), PNG)

    assert client_as(OUTSIDER).get(f"/people/{ALEX.id}/photo").status_code == 404
    assert client_as(SARAH).get(f"/people/{SARAH.id}/photo").status_code == 404  # none yet
    assert client_as(SARAH).get("/people/u-nobody/photo").status_code == 404


# team


def test_team_and_members_are_the_callers_own(client_as):
    sarah = client_as(SARAH)

    assert sarah.get("/team").json()["id"] == TEAM.id
    assert {p["id"] for p in sarah.get("/team/members").json()} == {ALEX.id, SARAH.id}
    assert [p["id"] for p in client_as(OUTSIDER).get("/team/members").json()] == [OUTSIDER.id]


# workspace settings


def settings_body(**changes) -> dict:
    body = {
        "team_id": TEAM.id,
        "github": {"repo": "acme/checkout", "ref": "main"},
        "jira": {"site": "acme.atlassian.net", "project": "DS"},
        "voice": "voice-1",
        "wake_phrase": "Hey Omni",
        "sensitivity": "quiet",
        "interrupt_minutes": 3,
        "who_can_allow": "host",
        "timezone": "America/Vancouver",
    }
    return body | changes


def test_settings_default_until_saved(client_as):
    settings = client_as(ALEX).get("/settings").json()

    assert settings["team_id"] == TEAM.id
    assert settings["sensitivity"] == "balanced"
    assert settings["interrupt_minutes"] == 5
    assert settings["who_can_allow"] == "everyone"
    assert settings["timezone"] == "UTC"


def test_the_admin_saves_settings_that_teammates_then_read(client_as):
    response = client_as(ALEX).put("/settings", json=settings_body())

    assert response.status_code == 200
    saved = client_as(SARAH).get("/settings").json()
    assert saved == response.json()
    assert saved["voice"] == "voice-1"
    assert saved["wake_phrase"] == "Hey Omni"
    assert (saved["sensitivity"], saved["interrupt_minutes"], saved["who_can_allow"]) == (
        "quiet",
        3,
        "host",
    )
    assert saved["timezone"] == "America/Vancouver"


def test_settings_are_always_saved_for_the_callers_own_team(client_as):
    olga_as_admin = OUTSIDER.model_copy(update={"is_admin": True})
    response = client_as(olga_as_admin).put("/settings", json=settings_body(voice="hijack"))

    assert response.status_code == 200
    assert response.json()["team_id"] != TEAM.id
    assert client_as(ALEX).get("/settings").json()["voice"] is None


def test_settings_keep_the_connectors_whatever_the_body_says(client_as):
    """Connectors change only at PUT /settings/connectors, which needs an admin."""
    connectors = {
        "github": [{"path": "acme/checkout", "ref": "main"}],
        "gitlab": [{"path": "acme/infra"}],
        "jira": {"site": "acme.atlassian.net", "project": "DS"},
    }
    assert client_as(ADMIN).put("/settings/connectors", json=connectors).status_code == 200
    body = settings_body(
        github={"repos": [{"path": "evil/repo", "connected": True, "files": 9000}]},
        gitlab={"projects": []},
        jira={"site": "evil.example", "project": "EV", "connected": True},
    )

    saved = client_as(SARAH).put("/settings", json=body).json()

    assert [r["path"] for r in saved["github"]["repos"]] == ["acme/checkout"]
    assert saved["github"]["repos"][0]["connected"] is False
    assert [p["path"] for p in saved["gitlab"]["projects"]] == ["acme/infra"]
    assert (saved["jira"]["site"], saved["jira"]["project"]) == ("acme.atlassian.net", "DS")
    assert client_as(ALEX).get("/settings").json() == saved


def test_settings_may_leave_the_connectors_out_or_send_the_old_shape(client_as):
    without = {k: v for k, v in settings_body().items() if k not in ("github", "jira")}

    assert client_as(SARAH).put("/settings", json=without).status_code == 200
    assert client_as(SARAH).put("/settings", json=settings_body()).status_code == 200


def test_blank_text_settings_are_cleared(client_as):
    saved = client_as(ALEX).put("/settings", json=settings_body(wake_phrase="   ", voice="")).json()

    assert saved["wake_phrase"] is None
    assert saved["voice"] is None


def test_invalid_settings_are_rejected(client_as):
    alex = client_as(ALEX)

    for bad in (
        {"interrupt_minutes": 0},
        {"interrupt_minutes": 16},
        {"sensitivity": "loud"},
        {"who_can_allow": "nobody"},
    ):
        assert alex.put("/settings", json=settings_body(**bad)).status_code == 422, bad
    assert alex.put("/settings", json=settings_body(interrupt_minutes=1)).status_code == 200
    assert alex.put("/settings", json=settings_body(interrupt_minutes=15)).status_code == 200


@pytest.mark.parametrize(
    "zone", ["Mars/Olympus_Mons", "", "PST", "america/vancouver", "../../etc/passwd", "Vancouver"]
)
def test_a_time_zone_that_is_not_an_iana_name_is_rejected(client_as, zone):
    alex = client_as(ALEX)

    response = alex.put("/settings", json=settings_body(timezone=zone))

    assert response.status_code == 422, zone
    assert "time zone" in response.json()["detail"]
    assert alex.get("/settings").json()["timezone"] == "UTC"


def test_a_time_zone_is_saved_trimmed_and_utc_is_accepted(client_as):
    alex = client_as(ALEX)

    assert (
        alex.put("/settings", json=settings_body(timezone=" Europe/Berlin ")).json()["timezone"]
        == "Europe/Berlin"
    )
    assert alex.put("/settings", json=settings_body(timezone="UTC")).json()["timezone"] == "UTC"
