from typing import Protocol

from contracts import Report

REPORT_STEPS = [
    "Finalising the transcript",
    "Writing the summary",
    "Extracting decisions",
    "Drafting tasks and owners",
    "Checking against past decisions",
    "Linking code references",
    "Indexing for search",
]


class PostMeetingPipeline(Protocol):
    """Runs once a meeting ends: summary, decisions, task drafts, past-decision links, indexing.

    Private chat is never an input. The first step waits for the worker to flush its last
    final segments (or a short grace period) before reading the transcript: segments are
    still accepted while the meeting is `processing`, and refused once the report exists.
    """

    async def run(self, meeting_id: str) -> Report: ...
