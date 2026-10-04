"""The shared contract: v0 design fields, and the TypeScript mirror kept in lockstep."""

import inspect
import re
import typing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import contracts
from contracts import (
    Agenda,
    AgendaItem,
    AgendaTrackRequest,
    AgendaTrackResponse,
    AgendaUpdate,
    AgentState,
    AskRequest,
    ConnectorStatus,
    CreateMeetingRequest,
    FactCheck,
    FactCheckRequest,
    FactCheckResponse,
    Meeting,
    Person,
    ReportProgress,
)

TS_DIR = Path(contracts.__file__).resolve().parents[2] / "ts"


def test_a_meeting_can_be_scheduled_and_old_payloads_still_parse():
    old = Meeting(
        id="m1",
        team_id="t1",
        title="Standup",
        status="live",
        code="abc",
        host_id="u1",
        participant_ids=[],
    )
    scheduled = old.model_copy(
        update={
            "status": "scheduled",
            "scheduled_start": datetime(2026, 10, 1, 16, tzinfo=UTC),
            "invitee_ids": ["u2"],
        }
    )

    assert (old.invitee_ids, old.scheduled_start, old.ended_at) == ([], None, None)
    assert old.transcript_deleted_at is None
    assert Meeting.model_validate(scheduled.model_dump()) == scheduled


def test_people_have_an_optional_email_and_photo():
    alex = Person(id="u1", name="Alex Chen", short="Alex", initials="AC")

    assert (alex.email, alex.photo_url) == (None, None)


def test_requests_from_before_the_v0_design_still_validate():
    assert CreateMeetingRequest(title="Standup").invitee_ids == []
    assert AskRequest(question="Is DS-117 done?", visibility="public").history == []
    assert ReportProgress(meeting_id="m1", steps=[], current=0, done=False).error is None


def test_a_follow_up_question_carries_the_earlier_turns():
    body = AskRequest.model_validate(
        {
            "question": "And who owns it?",
            "visibility": "private",
            "history": [
                {"role": "user", "text": "Is DS-117 done?"},
                {"role": "agent", "text": "No, it is in progress."},
            ],
        }
    )

    assert [t.role for t in body.history] == ["user", "agent"]
    with pytest.raises(ValidationError):
        AskRequest.model_validate(
            {"question": "q", "visibility": "public", "history": [{"role": "boss", "text": "x"}]}
        )


def test_an_agenda_edit_may_add_items_without_ids():
    update = AgendaUpdate.model_validate(
        {"items": [{"id": "a1", "title": "Refunds", "minutes": 10}, {"title": "Rollout"}]}
    )

    assert [(i.id, i.minutes) for i in update.items] == [("a1", 10), (None, None)]
    assert [i.status for i in update.items] == [None, None]


def test_an_agenda_edit_may_set_an_items_status():
    update = AgendaUpdate.model_validate(
        {"items": [{"id": "a1", "title": "Refunds", "status": "pending"}, {"title": "Docs"}]}
    )

    assert [i.status for i in update.items] == ["pending", None]
    with pytest.raises(ValidationError):
        AgendaUpdate.model_validate({"items": [{"title": "Refunds", "status": "done"}]})


def test_agendas_saved_before_timekeeping_still_parse():
    old = Agenda.model_validate(
        {
            "meeting_id": "m1",
            "items": [{"id": "a1", "title": "Refunds", "sources": [], "status": "pending"}],
            "generated_at": "2026-10-01T16:00:00Z",
        }
    )

    assert (old.current_item_id, old.tracked_until) == (None, None)
    assert (old.items[0].discussed_s, old.items[0].nudged_t) == (0, None)
    assert old.revision == 0


def test_a_track_request_may_omit_now_but_never_goes_negative():
    assert AgendaTrackRequest().now is None
    assert AgendaTrackRequest(now=90.5).now == 90.5
    for bad in (-1, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            AgendaTrackRequest(now=bad)
    with pytest.raises(ValidationError):
        AgendaItem(id="a1", title="Refunds", discussed_s=-1)
    response = AgendaTrackResponse(
        agenda=Agenda(meeting_id="m1", items=[], generated_at=datetime(2026, 10, 1, tzinfo=UTC)),
        nudges=[],
    )
    assert AgendaTrackResponse.model_validate(response.model_dump()) == response


def test_a_fact_check_tick_may_omit_now_and_returns_nothing_by_default():
    assert FactCheckRequest().now is None
    with pytest.raises(ValidationError):
        FactCheckRequest(now=-1)
    empty = FactCheckResponse()
    assert (empty.checks, empty.agent_state, empty.snippets) == ([], None, [])
    raised = FactCheckResponse(
        checks=[
            FactCheck(
                id="f1",
                claim="PR 41 is released.",
                speaker_name="Sarah Kim",
                verdict="contradicted",
                confidence=0.9,
                severity="high",
                raised_hand=True,
            )
        ],
        agent_state=AgentState(
            state="hand_raised", detail="Fact-check", hand_urgency="critical", hand_reason="x"
        ),
    )
    assert FactCheckResponse.model_validate(raised.model_dump()) == raised


def test_connector_status_names_only_known_connectors_and_states():
    ok = ConnectorStatus(name="github", state="connected")

    assert ok.detail is None
    with pytest.raises(ValidationError):
        ConnectorStatus(name="slack", state="connected")
    with pytest.raises(ValidationError):
        ConnectorStatus(name="jira", state="maybe")


# the TypeScript mirror


def ts_source() -> str:
    return "\n".join(p.read_text() for p in sorted(TS_DIR.glob("*.ts")))


def ts_interfaces() -> dict[str, dict[str, bool]]:
    """interface name -> {field: optional}"""
    found: dict[str, dict[str, bool]] = {}
    for name, body in re.findall(r"export interface (\w+) \{(.*?)\n\}", ts_source(), re.S):
        found[name] = {
            field: bool(optional) for field, optional in re.findall(r"^\s+(\w+)(\??):", body, re.M)
        }
    return found


def ts_string_unions() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for name, union in re.findall(r"export type (\w+) =([^;]+);", ts_source()):
        found[name] = set(re.findall(r'"([^"]*)"', union))
    return found


def contract_models() -> dict[str, type[BaseModel]]:
    return {
        name: obj
        for name in contracts.__all__
        if inspect.isclass(obj := getattr(contracts, name)) and issubclass(obj, BaseModel)
    }


def contract_literals() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for module in {inspect.getmodule(m) for m in contract_models().values()}:
        for name, value in vars(module).items():
            if typing.get_origin(value) is typing.Literal:
                found[name] = set(typing.get_args(value))
    return found


def test_every_exported_model_has_a_typescript_interface_with_the_same_fields():
    interfaces = ts_interfaces()

    for name, model in contract_models().items():
        assert name in interfaces, f"contracts/ts has no interface {name}"
        ts_fields = interfaces[name]
        assert set(ts_fields) == set(model.model_fields), name
        for field, info in model.model_fields.items():
            if info.is_required():
                assert not ts_fields[field], f"{name}.{field} is required in Python"
            elif info.default is None:
                assert ts_fields[field], f"{name}.{field} is optional in Python"


def test_every_literal_type_has_the_same_values_in_typescript():
    unions = ts_string_unions()

    for name, values in contract_literals().items():
        assert unions.get(name) == values, name


def test_typescript_exports_every_module():
    index = (TS_DIR / "index.ts").read_text()

    for module in TS_DIR.glob("*.ts"):
        if module.stem != "index":
            assert f'export * from "./{module.stem}";' in index
