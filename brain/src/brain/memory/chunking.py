"""Final transcripts and reports into chunks. Private chat has no way in, and neither do the
agent's own words: they are never evidence."""

from collections.abc import Iterable

from brain.speakers import by_agent
from contracts import Report, TranscriptSegment

from .models import Chunk

MAX_CHUNK_CHARS = 1200
"""About a minute and a half of conversation: an hour-long meeting makes 30-odd chunks, so the
free embedding tier's daily texts cover the demo's meetings, and each chunk still carries enough
of the exchange around a fact to be found by it. Well under the embedding models' input limits."""

Turn = list[TranscriptSegment]


def chunk_transcript(
    team_id: str,
    meeting_id: str,
    segments: Iterable[TranscriptSegment],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Windows of consecutive turns, each turn on its own `Speaker: text` line, packed up to
    `max_chars`. A turn longer than that is split between segments; a single segment longer
    than that stays whole. The agent's turns are left out."""
    windows: list[list[Turn]] = []
    for turn in people_turns(meeting_id, segments):
        for i, segment in enumerate(turn):
            window = windows[-1] if windows else None
            if window is None:
                windows.append([[segment]])
                continue
            # The turn's first segment starts a new line; later ones continue the last line.
            grown = [*window, [segment]] if i == 0 else [*window[:-1], [*window[-1], segment]]
            if len(window_text(grown)) <= max_chars:
                window[:] = grown
            else:
                windows.append([[segment]])
    return [window_chunk(team_id, meeting_id, window) for window in windows]


def people_turns(meeting_id: str, segments: Iterable[TranscriptSegment]) -> list[Turn]:
    """Runs of consecutive final segments by one speaker, in time order, without the agent's.
    The agent speaking between two lines of one person ends that person's turn."""
    turns: list[Turn] = []
    previous: TranscriptSegment | None = None
    for segment in final_segments(meeting_id, segments):
        if not by_agent(segment):
            if previous is not None and previous.speaker_id == segment.speaker_id:
                turns[-1].append(segment)
            else:
                turns.append([segment])
        previous = segment
    return turns


def window_chunk(team_id: str, meeting_id: str, window: list[Turn]) -> Chunk:
    first, last = window[0][0], window[-1][-1]
    speakers = {turn[0].speaker_id for turn in window}
    alone = len(speakers) == 1
    return Chunk(
        id=f"{meeting_id}:transcript:{first.seg_id}",
        team_id=team_id,
        meeting_id=meeting_id,
        kind="transcript",
        text=window_text(window),
        speaker_id=first.speaker_id if alone else None,
        speaker_name=first.speaker_name if alone else None,
        t_start=first.t_start,
        t_end=last.t_end,
    )


def final_segments(
    meeting_id: str, segments: Iterable[TranscriptSegment]
) -> list[TranscriptSegment]:
    seen: set[str] = set()
    final = []
    for segment in sorted(segments, key=lambda s: (s.t_start, s.seg_id)):
        if segment.meeting_id != meeting_id:
            raise ValueError(f"segment {segment.seg_id} belongs to meeting {segment.meeting_id}")
        if segment.is_final and segment.seg_id not in seen:
            seen.add(segment.seg_id)
            final.append(segment)
    return final


def window_text(window: list[Turn]) -> str:
    return "\n".join(turn_text(turn) for turn in window)


def turn_text(run: Turn) -> str:
    return f"{run[0].speaker_name}: " + " ".join(s.text.strip() for s in run)


def chunk_report(team_id: str, report: Report) -> list[Chunk]:
    """The summary, each decision and each task draft, one chunk each."""
    meeting_id = report.meeting_id
    chunks: list[Chunk] = []
    if report.summary.strip():
        chunks.append(
            Chunk(
                id=f"{meeting_id}:summary",
                team_id=team_id,
                meeting_id=meeting_id,
                kind="summary",
                text=f"Summary: {report.summary.strip()}",
            )
        )
    for decision in report.decisions:
        chunks.append(
            Chunk(
                id=f"{meeting_id}:decision:{decision.id}",
                team_id=team_id,
                meeting_id=meeting_id,
                kind="decision",
                text=f"Decision: {decision.text}",
                speaker_id=decision.made_by,
                ref_id=decision.id,
                t_start=decision.t,
            )
        )
    for task in report.tasks:
        text = f"Task: {task.title}"
        if task.description:
            text += f"\n{task.description}"
        chunks.append(
            Chunk(
                id=f"{meeting_id}:task:{task.id}",
                team_id=team_id,
                meeting_id=meeting_id,
                kind="task",
                text=text,
                ref_id=task.id,
                t_start=task.t,
            )
        )
    return chunks
