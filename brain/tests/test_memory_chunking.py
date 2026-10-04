from itertools import pairwise
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


def test_consecutive_turns_pack_into_one_window_a_line_each():
    segments = [
        seg(1, "Bob", "The fix is merged.", 0),
        seg(2, "Bob", "Refunds go out Wednesday.", 5),
        seg(3, "Carol", "Production is on v0.9.3.", 10),
        seg(4, "Bob", "Then we wait.", 15),
    ]

    (chunk,) = chunk_transcript("t-1", "m-1", segments)

    assert chunk.text == (
        "Bob: The fix is merged. Refunds go out Wednesday.\n"
        "Carol: Production is on v0.9.3.\n"
        "Bob: Then we wait."
    )
    assert (chunk.t_start, chunk.t_end) == (0, 19)
    assert (chunk.kind, chunk.team_id, chunk.meeting_id) == ("transcript", "t-1", "m-1")


def lines(chunks) -> list[str]:
    return [line for c in chunks for line in c.text.splitlines()]


def test_a_window_holds_whole_turns_up_to_the_size_cap():
    speakers = ["Alice", "Bob", "Carol"]
    segments = [seg(i, speakers[i % 3], f"Turn {i} says a few words.", i * 5) for i in range(30)]

    chunks = chunk_transcript("t-1", "m-1", segments, max_chars=100)

    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    # every turn whole, on its own line, once and in order
    assert lines(chunks) == [f"{s.speaker_name}: {s.text}" for s in segments]
    # no window could have taken the next window's first line
    for window, following in pairwise(chunks):
        assert len(window.text + "\n" + following.text.splitlines()[0]) > 100
    assert [(c.t_start, c.t_end) for c in chunks] == sorted((c.t_start, c.t_end) for c in chunks)
    assert chunks[0].t_start == 0 and chunks[-1].t_end == 29 * 5 + 4


def test_a_turn_longer_than_a_window_is_split_between_segments():
    segments = [
        seg(1, "Alice", "Go ahead.", 0),
        *(seg(i, "Bob", f"part {i} " + "word " * 15, i * 5) for i in range(2, 8)),
        seg(8, "Carol", "Thanks.", 40),
    ]

    chunks = chunk_transcript("t-1", "m-1", segments, max_chars=250)

    assert len(chunks) > 2
    assert all(len(c.text) <= 250 for c in chunks)
    bob = [line for line in lines(chunks) if line.startswith("Bob: ")]
    assert len(bob) == len(chunks)  # one piece of Bob's turn per window, each with his name
    assert " ".join(line.removeprefix("Bob: ") for line in bob) == " ".join(
        s.text.strip() for s in segments[1:-1]
    )
    assert lines(chunks)[0] == "Alice: Go ahead."
    assert lines(chunks)[-1] == "Carol: Thanks."


def test_a_single_segment_over_the_cap_stays_whole_in_its_own_window():
    segments = [seg(1, "Alice", "a", 0), seg(2, "Bob", "x" * 400, 5), seg(3, "Carol", "c", 10)]

    chunks = chunk_transcript("t-1", "m-1", segments, max_chars=100)

    assert [c.text for c in chunks] == ["Alice: a", "Bob: " + "x" * 400, "Carol: c"]


def test_only_final_segments_count_once_each_in_time_order():
    segments = [
        seg(2, "Carol", "later", 10),
        seg(1, "Bob", "partial guess", 0, final=False),
        seg(1, "Bob", "final words", 0),
        seg(1, "Bob", "final words", 0),
    ]

    chunks = chunk_transcript("t-1", "m-1", segments)

    assert [c.text for c in chunks] == ["Bob: final words\nCarol: later"]


def test_segments_from_another_meeting_are_refused():
    with pytest.raises(ValueError, match="m-2"):
        chunk_transcript("t-1", "m-1", [seg(1, "Bob", "a", 0, meeting="m-2")])


def test_a_window_of_one_speaker_carries_them_and_a_mixed_one_carries_no_one():
    alone = chunk_transcript("t-1", "m-1", [seg(1, "Bob", "a", 0), seg(2, "Bob", "b", 5)])
    mixed = chunk_transcript("t-1", "m-1", [seg(1, "Bob", "a", 0), seg(2, "Carol", "b", 5)])

    assert [(c.speaker_id, c.speaker_name) for c in alone] == [("bob", "Bob")]
    assert [(c.speaker_id, c.speaker_name) for c in mixed] == [(None, None)]


def test_chunk_ids_are_the_first_segment_of_each_window_stable_and_unique():
    segments = [seg(i, ["Bob", "Carol"][i % 2], "word " * 10, i * 5) for i in range(1, 13)]

    first = chunk_transcript("t-1", "m-1", segments, max_chars=120)
    again = chunk_transcript("t-1", "m-1", list(reversed(segments)), max_chars=120)

    assert [c.id for c in first] == [c.id for c in again]
    assert len({c.id for c in first}) == len(first) > 1
    assert first[0].id == "m-1:transcript:s1"
    for chunk in first:
        (opening,) = [s for s in segments if s.t_start == chunk.t_start]
        assert chunk.id == f"m-1:transcript:{opening.seg_id}"


def agent_seg(n: int, text: str, t: float) -> TranscriptSegment:
    return seg(n, "Bob", text, t).model_copy(
        update={"speaker_id": AGENT_PARTICIPANT_ID, "speaker_name": get_identity().agent_name}
    )


def test_the_agents_own_words_are_never_chunked():
    segments = [
        seg(1, "Bob", "Is DS-104 done?", 0),
        agent_seg(2, "DS-104 is still In Progress in Jira.", 5),
        agent_seg(3, "The fix is merged though.", 10),
        seg(4, "Carol", "Then I'll close it.", 15),
    ]

    (chunk,) = chunk_transcript("t-1", "m-1", segments)

    assert chunk.text == "Bob: Is DS-104 done?\nCarol: Then I'll close it."
    assert (chunk.speaker_id, chunk.t_start, chunk.t_end) == (None, 0, 19)


def test_the_agent_between_two_lines_of_one_person_leaves_two_turns():
    segments = [
        seg(1, "Bob", "Is DS-104 done?", 0),
        agent_seg(2, "DS-104 is still In Progress in Jira.", 5),
        seg(3, "Bob", "Then I'll close it.", 10),
    ]

    (chunk,) = chunk_transcript("t-1", "m-1", segments)

    assert chunk.text == "Bob: Is DS-104 done?\nBob: Then I'll close it."
    assert (chunk.speaker_id, chunk.speaker_name) == ("bob", "Bob")


def test_the_agent_never_opens_closes_or_sizes_a_window():
    long_answer = "In Progress. " * 40
    segments = [
        agent_seg(1, "Hello, I'm listening.", 0),
        seg(2, "Bob", "Is DS-104 done?", 5),
        agent_seg(3, long_answer, 10),
        seg(4, "Carol", "Then I'll close it.", 60),
        agent_seg(5, "Noted.", 65),
    ]

    (chunk,) = chunk_transcript("t-1", "m-1", segments, max_chars=100)

    assert chunk.text == "Bob: Is DS-104 done?\nCarol: Then I'll close it."
    assert (chunk.id, chunk.t_start, chunk.t_end) == ("m-1:transcript:s2", 5, 64)


def test_a_transcript_of_only_the_agent_makes_no_chunks():
    assert chunk_transcript("t-1", "m-1", [agent_seg(1, "Hello, I'm listening.", 0)]) == []


def test_the_standup_is_one_window_of_the_peoples_turns():
    meeting = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())

    (chunk,) = chunk_transcript("t-1", meeting.meeting_id, meeting.segments)

    assert [line.split(": ")[0] for line in chunk.text.splitlines()] == [
        "Alice Moreau",
        "Bob Okafor",
        "Carol Jensen",
        "Alice Moreau",
        "Carol Jensen",
    ]
    assert "Bob Okafor: The double-charge fix is merged." in chunk.text
    assert "In Progress" not in chunk.text
    assert (chunk.speaker_id, chunk.t_start, chunk.t_end) == (None, 0, 47)


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
