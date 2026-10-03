"""World records, shaped like the GitHub and Jira API objects the real MCP servers return."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

from contracts import ChatMessage, TranscriptSegment


class GitHubUser(BaseModel):
    login: str


class GitHubComment(BaseModel):
    id: int
    user: GitHubUser
    body: str
    created_at: datetime


class GitHubIssue(BaseModel):
    number: int
    title: str
    body: str
    state: Literal["open", "closed"]
    user: GitHubUser
    labels: list[str] = []
    assignees: list[GitHubUser] = []
    comments: list[GitHubComment] = []
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    html_url: str


class GitHubPullRequest(BaseModel):
    number: int
    title: str
    body: str
    state: Literal["open", "closed"]
    merged: bool
    merged_at: datetime | None = None
    user: GitHubUser
    head: str
    base: str
    closes: list[int] = []
    comments: list[GitHubComment] = []
    created_at: datetime
    updated_at: datetime
    html_url: str


class GitHubCommit(BaseModel):
    sha: str
    message: str
    author: GitHubUser
    date: datetime
    html_url: str


class GitHubRelease(BaseModel):
    tag_name: str
    name: str
    body: str
    target_commitish: str
    published_at: datetime
    html_url: str


class JiraComment(BaseModel):
    id: str
    author: str
    body: str
    created: datetime


class JiraIssue(BaseModel):
    key: str
    summary: str
    description: str | None = None
    issuetype: str
    status: str
    assignee: str | None = None
    priority: str | None = None
    duedate: date | None = None
    sprint: str | None = None
    parent: str | None = None
    comments: list[JiraComment] = []
    created: datetime
    updated: datetime


class PastMeeting(BaseModel):
    """Seeded into the brain's own memory store, the same place real meetings go."""

    id: str
    title: str
    started_at: datetime
    participant_ids: list[str]
    segments: list[TranscriptSegment]
    public_chat: list[ChatMessage] = []


class Snapshot(BaseModel):
    name: str
    issues: list[GitHubIssue]
    pull_requests: list[GitHubPullRequest]
    commits: list[GitHubCommit]
    releases: list[GitHubRelease]
    jira_issues: list[JiraIssue]
    meetings: list[PastMeeting]
