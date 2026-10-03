from datetime import datetime
from pathlib import Path

from brain.report import TranscriptInput, load_transcript, parse_text_transcript
from contracts import AGENT_PARTICIPANT_ID, get_identity

FIXTURES = Path(__file__).parent / "fixtures"


def test_json_transcripts_load_as_is():
    meeting = load_transcript(FIXTURES / "standup.json")

    assert isinstance(meeting, TranscriptInput)
    assert meeting.meeting_id == "mtg-standup"
    assert len(meeting.segments) == 6
    assert [p.id for p in meeting.members] == ["p-alice", "p-bob", "p-carol"]


def test_text_transcripts_become_segments():
    agent = get_identity().agent_name
    text = f"""# Friday standup

[00:00] Alice Moreau: Morning everyone.
[00:06] Bob Okafor: The fix is merged.
I'll refund the users by Wednesday.
Carol Jensen: The exploit fix is not deployed.

[1:02:03] {agent}: DS-104 is stale.
"""

    meeting = parse_text_transcript(text, meeting_id="standup")

    assert meeting.title == "Friday standup"
    assert [s.speaker_id for s in meeting.segments] == [
        "alice-moreau",
        "bob-okafor",
        "carol-jensen",
        AGENT_PARTICIPANT_ID,
    ]
    bob, carol, said_by_agent = meeting.segments[1:]
    assert bob.text == "The fix is merged. I'll refund the users by Wednesday."
    assert (bob.t_start, bob.t_end) == (6, 6)  # Carol's line has no timestamp of its own
    assert carol.t_start == 6
    assert said_by_agent.t_start == 3723
    assert said_by_agent.t_end == said_by_agent.t_start
    assert meeting.segments[0].t_end == 6
    assert all(s.is_final and s.meeting_id == "standup" for s in meeting.segments)
    assert len({s.seg_id for s in meeting.segments}) == 4


def test_people_are_derived_from_speakers_without_the_agent():
    meeting = parse_text_transcript(
        f"Alice Moreau: Hi.\n{get_identity().agent_name}: Hello.\nbob: Hey.", meeting_id="m"
    )

    people = meeting.people()

    assert [(p.id, p.name, p.short, p.initials) for p in people] == [
        ("alice-moreau", "Alice Moreau", "Alice", "AM"),
        ("bob", "bob", "bob", "B"),
    ]


def test_text_files_take_their_id_from_the_name_and_the_date_from_the_caller(tmp_path):
    path = tmp_path / "retro-oct-1.txt"
    path.write_text("Alice Moreau: Let's begin.\n")

    meeting = load_transcript(path, started_at=datetime(2026, 10, 1, 16, 0))

    assert meeting.meeting_id == "retro-oct-1"
    assert meeting.title == "retro-oct-1"
    assert meeting.started_at == datetime(2026, 10, 1, 16, 0)
