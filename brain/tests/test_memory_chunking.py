from pathlib import Path

import pytest

from brain.memory import chunk_report, chunk_transcript
from brain.report import TranscriptInput
from contracts import (
    AGENT_PARTICIPANT_ID,
    Decision,
    Report,
    TaskDraft,
    TranscriptSegment,
    get_identity,
)

FIXTURES = Path(__file__).parent / "fixtures"


def seg(
    n: int, speaker: str, text: str, t: float, *, final=True, meeting="m-1"
) -> TranscriptSegment:
    return TranscriptSegment(
        seg_id=f"s{n}",
        meeting_id=meeting,
        speaker_id=speaker.lower(),
        speaker_name=speaker,
        text=text,
        is_final=final,
        t_start=t,
        t_end=t + 4,
    )


def test_consecutive_turns_by_one_speaker_form_one_chunk():
    segments = [
        seg(1, "Bob", "The fix is merged.", 0),
        seg(2, "Bob", "Refunds go out Wednesday.", 5),
        seg(3, "Carol", "Production is on v0.9.3.", 10),
    ]

    chunks = chunk_transcript("t-1", "m-1", segments)

    assert [(c.speaker_id, c.speaker_name, c.t_start, c.t_end) for c in chunks] == [
        ("bob", "Bob", 0, 9),
        ("carol", "Carol", 10, 14),
    ]
    assert chunks[0].text == "Bob: The fix is merged. Refunds go out Wednesday."
    assert all(c.kind == "transcript" for c in chunks)
    assert all((c.team_id, c.meeting_id) == ("t-1", "m-1") for c in chunks)


def test_a_returning_speaker_starts_a_new_chunk():
    segments = [seg(1, "Bob", "a", 0), seg(2, "Carol", "b", 5), seg(3, "Bob", "c", 10)]

    chunks = chunk_transcript("t-1", "m-1", segments)

    assert [c.speaker_id for c in chunks] == ["bob", "carol", "bob"]


def test_a_long_turn_is_split_at_the_size_cap():
    segments = [seg(i, "Bob", "word " * 20, i * 5) for i in range(1, 7)]

    chunks = chunk_transcript("t-1", "m-1", segments, max_chars=250)

    assert len(chunks) > 1
    assert all(len(c.text) <= 250 for c in chunks)
    assert chunks[0].t_start == 5 and chunks[-1].t_end == 34
    assert [c.t_start for c in chunks] == sorted(c.t_start for c in chunks)


def test_a_single_segment_over_the_cap_stays_whole():
    chunks = chunk_transcript("t-1", "m-1", [seg(1, "Bob", "x" * 400, 0)], max_chars=100)

    assert len(chunks) == 1
    assert chunks[0].text.endswith("x" * 400)


def test_only_final_segments_count_once_each_in_time_order():
    segments = [
        seg(2, "Carol", "later", 10),
        seg(1, "Bob", "partial guess", 0, final=False),
        seg(1, "Bob", "final words", 0),
        seg(1, "Bob", "final words", 0),
    ]

    chunks = chunk_transcript("t-1", "m-1", segments)

    assert [c.text for c in chunks] == ["Bob: final words", "Carol: later"]


def test_segments_from_another_meeting_are_refused():
    with pytest.raises(ValueError, match="m-2"):
        chunk_transcript("t-1", "m-1", [seg(1, "Bob", "a", 0, meeting="m-2")])


def test_chunk_ids_are_stable_and_unique():
    segments = [seg(1, "Bob", "a", 0), seg(2, "Carol", "b", 5), seg(3, "Bob", "c", 10)]

    first = chunk_transcript("t-1", "m-1", segments)
    again = chunk_transcript("t-1", "m-1", list(reversed(segments)))

    assert [c.id for c in first] == [c.id for c in again]
    assert len({c.id for c in first}) == 3


def agent_seg(n: int, text: str, t: float) -> TranscriptSegment:
    return seg(n, "Bob", text, t).model_copy(
        update={"speaker_id": AGENT_PARTICIPANT_ID, "speaker_name": get_identity().agent_name}
    )


def test_the_agents_own_words_are_never_chunked():
    segments = [
        seg(1, "Bob", "Is DS-104 done?", 0),
        agent_seg(2, "DS-104 is still In Progress in Jira.", 5),
        agent_seg(3, "The fix is merged though.", 10),
        seg(4, "Bob", "Then I'll close it.", 15),
    ]

    chunks = chunk_transcript("t-1", "m-1", segments)

    assert [c.text for c in chunks] == ["Bob: Is DS-104 done?", "Bob: Then I'll close it."]
    assert all(c.speaker_id != AGENT_PARTICIPANT_ID for c in chunks)


def test_a_transcript_of_only_the_agent_makes_no_chunks():
    assert chunk_transcript("t-1", "m-1", [agent_seg(1, "Hello, I'm listening.", 0)]) == []


def test_the_standup_has_one_chunk_per_turn():
    meeting = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())

    chunks = chunk_transcript("t-1", meeting.meeting_id, meeting.segments)

    assert [c.speaker_id for c in chunks] == [
        "p-alice",
        "p-bob",
        "p-carol",
        "p-alice",
        "p-carol",
    ]
    assert not any("In Progress" in c.text for c in chunks)
    bob = chunks[1]
    assert bob.text.startswith("Bob Okafor: ")
    assert "refund the 14 affected users" in bob.text
    assert (bob.t_start, bob.t_end) == (6, 14)


def report(**updates) -> Report:
    base = Report(
        meeting_id="m-1",
        summary="Refunds go out Wednesday; the waitlist email waits for v0.9.4.",
        decisions=[
            Decision(
                id="d-1",
                meeting_id="m-1",
                text="Hold the waitlist email until v0.9.4 ships",
                made_by="alice",
                t=24,
                quote="Then we hold the waitlist email until v0.9.4 is out.",
            )
        ],
        tasks=[
            TaskDraft(
                id="task-1",
                meeting_id="m-1",
                title="Refund the 14 double-charged users",
                description="Bob refunds by Wednesday.",
                owner_id="bob",
                t=6,
            ),
            TaskDraft(id="task-2", meeting_id="m-1", title="Own the model retirement"),
        ],
    )
    return base.model_copy(update=updates)


def test_a_report_becomes_summary_decision_and_task_chunks():
    chunks = chunk_report("t-1", report())

    assert [(c.kind, c.ref_id) for c in chunks] == [
        ("summary", None),
        ("decision", "d-1"),
        ("task", "task-1"),
        ("task", "task-2"),
    ]
    assert all((c.team_id, c.meeting_id) == ("t-1", "m-1") for c in chunks)
    summary, decision, refund, retirement = chunks
    assert "Refunds go out Wednesday" in summary.text
    assert "Hold the waitlist email until v0.9.4 ships" in decision.text
    assert (decision.speaker_id, decision.t_start) == ("alice", 24)
    assert "Refund the 14 double-charged users" in refund.text
    assert "Bob refunds by Wednesday." in refund.text
    assert refund.t_start == 6
    assert retirement.t_start is None
    assert len({c.id for c in chunks}) == 4


def test_an_empty_summary_is_not_indexed():
    chunks = chunk_report("t-1", report(summary="  ", decisions=[], tasks=[]))

    assert chunks == []
