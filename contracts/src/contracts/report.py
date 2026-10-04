from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

from .agent import CodeSnippet, FactCheck, QuestionAnswered, Severity, Source

DecisionStatus = Literal["active", "superseded"]
DecisionRelationType = Literal["contradicts", "superseded_by"]
JiraStatus = Literal["draft", "todo", "in_progress", "done"]
TaskDestination = Literal["jira", "github"]


class DecisionRelation(BaseModel):
    type: DecisionRelationType
    decision_id: str


class DecisionStep(BaseModel):
    """One thing said on the way to a decision: a few words on what, and where it was said."""

    text: str
    t: float  # seconds from the meeting start, where this part of the conversation begins
    seg_ids: list[str] = []  # the transcript segments it describes, in time order


class Decision(BaseModel):
    id: str
    meeting_id: str
    text: str
    made_by: str
    t: float
    quote: str
    # What was said that led to it, oldest first. Empty when none was recorded: the decision
    # then has only its quote.
    chain: list[DecisionStep] = []
    status: DecisionStatus = "active"
    relation: DecisionRelation | None = None


class TaskDraft(BaseModel):
    """Internal task draft. Owner and due date only when stated or clearly inferable."""

    id: str
    meeting_id: str
    title: str
    description: str | None = None
    owner_id: str | None = None
    due: date | None = None
    t: float | None = None
    quote: str | None = None
    include: bool = True
    key: str | None = None  # external key once pushed
    jira_status: JiraStatus = "draft"


class Risk(BaseModel):
    text: str
    severity: Severity


class Report(BaseModel):
    """Post-meeting write-up. Empty sections stay empty; nothing is invented to fill them."""

    meeting_id: str
    summary: str
    topics: list[str] = []
    decisions: list[Decision] = []
    tasks: list[TaskDraft] = []
    risks: list[Risk] = []
    open_questions: list[str] = []
    blockers: list[str] = []
    technical_context: list[str] = []
    links: list[Source] = []
    fact_checks: list[FactCheck] = []
    questions: list[QuestionAnswered] = []
    snippets: list[CodeSnippet] = []


class ReportProgress(BaseModel):
    meeting_id: str
    steps: list[str]
    current: int
    done: bool
    error: str | None = None  # why the pipeline stopped at the current step
    updated_at: datetime | None = None  # when the write-up last saved progress


class TaskPushRequest(BaseModel):
    """A specific human approval: which drafts go where."""

    task_ids: list[str]
    destination: TaskDestination
    approved_by: str


class TaskPushResult(BaseModel):
    task_id: str
    key: str | None = None
    url: str | None = None
    error: str | None = None
    warning: str | None = None  # e.g. "created unassigned: ..." when it was created anyway
