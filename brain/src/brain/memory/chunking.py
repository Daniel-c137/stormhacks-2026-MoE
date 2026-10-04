"""Final transcripts and reports into chunks. Private chat has no way in, and neither do the
agent's own words: they are never evidence."""

from collections.abc import Iterable

from brain.speakers import by_agent
from contracts import Report, TranscriptSegment

from .models import Chunk

MAX_CHUNK_CHARS = 1200


def chunk_transcript(
    team_id: str,
    meeting_id: str,
    segments: Iterable[TranscriptSegment],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """One chunk per run of consecutive final segments by the same speaker, split when it would
    grow past `max_chars`. A single segment longer than that stays whole. The agent's turns
    are left out."""
    final = final_segments(meeting_id, segments)
    runs: list[list[TranscriptSegment]] = []
    for segment in final:
        run = runs[-1] if runs else None
        if (
            run
            and run[-1].speaker_id == segment.speaker_id
            and len(turn_text([*run, segment])) <= max_chars
        ):
            run.append(segment)
        else:
            runs.append([segment])
    return [
        Chunk(
            id=f"{meeting_id}:transcript:{run[0].seg_id}",
            team_id=team_id,
            meeting_id=meeting_id,
            kind="transcript",
            text=turn_text(run),
            speaker_id=run[0].speaker_id,
            speaker_name=run[0].speaker_name,
            t_start=run[0].t_start,
            t_end=run[-1].t_end,
        )
        for run in runs
        if not by_agent(run[0])
    ]


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


def turn_text(run: list[TranscriptSegment]) -> str:
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
