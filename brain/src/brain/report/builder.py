from collections.abc import Sequence
from datetime import UTC, date, tzinfo

from brain.llm import LLM
from brain.zones import local_date
from contracts import (
    AGENT_PARTICIPANT_ID,
    AgendaItem,
    Decision,
    FactCheck,
    Report,
    Risk,
    Source,
    TaskDraft,
    TranscriptSegment,
    get_identity,
)

from .extraction import ReportExtraction, render_prompt, system_prompt
from .models import TranscriptInput


async def build_report(
    llm: LLM,
    meeting: TranscriptInput,
    *,
    agenda: Sequence[AgendaItem] = (),
    zone: tzinfo = UTC,
    fact_checks: Sequence[FactCheck] = (),
) -> Report:
    """Ask the LLM for the meeting record, then keep only what the transcript supports. With an
    agenda, the topics follow its items in order. The meeting's date, which relative due dates
    resolve against, is its day in the team's `zone`. The meeting's fact-checks that confidently
    contradicted a claim go to the model, so the record never states that claim as fact."""
    segments = meeting.final_segments()
    if not segments:
        raise ValueError(f"Meeting {meeting.meeting_id} has no final segments to report on")
    people = meeting.people()
    labelled = {f"s{i}": s for i, s in enumerate(segments, 1)}
    meeting_date = local_date(meeting.started_at, zone)
    day = f"{meeting_date.isoformat()} ({meeting_date:%A})" if meeting_date else None

    extraction = await llm.generate_structured(
        render_prompt(
            title=meeting.title,
            meeting_date=day,
            people=people,
            labelled=labelled,
            agenda=agenda,
            fact_checks=fact_checks,
            agent_attended=meeting.agent_attended(),
        ),
        ReportExtraction,
        system=system_prompt(),
    )
    return Grounding(meeting, labelled, {p.id: p.name for p in people}, meeting_date).report(
        extraction
    )


class Grounding:
    """Quotes and times come from the cited segments, never from the model."""

    def __init__(
        self,
        meeting: TranscriptInput,
        labelled: dict[str, TranscriptSegment],
        names: dict[str, str],
        meeting_date: date | None,
    ):
        self.meeting_id = meeting.meeting_id
        self.labelled = labelled
        self.names = names
        self.meeting_date = meeting_date
        self.text = "\n".join(s.text for s in labelled.values()).casefold()

    def report(self, x: ReportExtraction) -> Report:
        return Report(
            meeting_id=self.meeting_id,
            summary=x.summary.strip(),
            topics=clean(x.topics),
            decisions=self.decisions(x),
            tasks=self.tasks(x),
            risks=[
                Risk(text=r.text.strip(), severity=r.severity)
                for r in x.risks
                if self.evidence(r.evidence)
            ],
            open_questions=clean(x.open_questions),
            blockers=clean(x.blockers),
            technical_context=clean(x.technical_context),
            links=[
                Source(kind=link.kind, label=link.label, url=link.url)
                for link in x.links
                if self.mentioned(link.label, link.url)
            ],
        )

    def tasks(self, x: ReportExtraction) -> list[TaskDraft]:
        drafts: list[TaskDraft] = []
        for task in x.tasks:
            if not (cited := self.evidence(task.evidence)):
                continue
            drafts.append(
                TaskDraft(
                    id=f"{self.meeting_id}-task-{len(drafts) + 1}",
                    meeting_id=self.meeting_id,
                    title=task.title.strip(),
                    description=(task.description or "").strip() or None,
                    owner_id=task.owner_id if task.owner_id in self.names else None,
                    due=self.due(task.due),
                    t=cited[0].t_start,
                    quote=cited[0].text,
                )
            )
        return drafts

    def decisions(self, x: ReportExtraction) -> list[Decision]:
        decisions: list[Decision] = []
        for decision in x.decisions:
            if not (cited := self.evidence(decision.evidence)):
                continue
            decisions.append(
                Decision(
                    id=f"{self.meeting_id}-decision-{len(decisions) + 1}",
                    meeting_id=self.meeting_id,
                    text=decision.text.strip(),
                    made_by=self.names.get(decision.made_by_id or "") or said_by(cited[0]),
                    t=cited[0].t_start,
                    quote=cited[0].text,
                )
            )
        return decisions

    def evidence(self, labels: list[str]) -> list[TranscriptSegment]:
        cited = dict.fromkeys(label.strip() for label in labels)
        return [self.labelled[label] for label in cited if label in self.labelled]

    def due(self, value: str | None) -> date | None:
        try:
            due = date.fromisoformat((value or "").strip())
        except ValueError:
            return None
        if self.meeting_date and due < self.meeting_date:
            return None
        return due

    def mentioned(self, label: str, url: str | None) -> bool:
        return label.strip().casefold() in self.text or bool(url and url.casefold() in self.text)


def said_by(segment: TranscriptSegment) -> str:
    if segment.speaker_id == AGENT_PARTICIPANT_ID:
        return get_identity().agent_name
    return segment.speaker_name


def clean(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]
