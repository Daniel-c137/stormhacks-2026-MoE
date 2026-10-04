export type AgentStateName = "idle" | "capturing" | "working" | "hand_raised" | "speaking" | "followup";
export type HandUrgency = "normal" | "critical";
export type Visibility = "public" | "private";
export type InvocationVia = "voice" | "follow_up" | "chat" | "ask" | "private";
export type SourceKind =
  | "meeting"
  | "github_issue"
  | "github_pr"
  | "github_code"
  | "github_release"
  | "gitlab_issue"
  | "gitlab_mr"
  | "gitlab_code"
  | "gitlab_release"
  | "jira_issue";
export type Verdict = "supported" | "contradicted" | "unknown";
export type Severity = "low" | "high";

export interface AgentState {
  state: AgentStateName;
  detail: string;
  hand_urgency: HandUrgency;
  hand_reason: string;
}

/** A deliberate request to the agent. The agent never reasons on every utterance. */
export interface Invocation {
  id: string;
  meeting_id: string;
  via: InvocationVia;
  visibility: Visibility;
  asked_by_id: string;
  asked_by_name: string;
  question: string;
  t?: number | null;
}

/** Evidence behind an answer: a meeting moment, a GitHub or GitLab item or a Jira key. A
 * repository's items name it: dropsubs/web#41, group/project!12 (a GitLab merge request). */
export interface Source {
  kind: SourceKind;
  label: string;
  url?: string | null;
  meeting_id?: string | null;
  t?: number | null;
}

export interface CodeSnippet {
  id: string;
  path: string;
  start_line: number;
  end_line: number;
  language: string;
  code: string;
  github_url: string; // the lines on their code host: GitHub, or GitLab for a GitLab project
  caption: string;
  repo?: string | null; // owner/name on GitHub, the project path on GitLab
  highlight?: [number, number] | null; // UI-only
}

export interface Answer {
  id: string;
  invocation_id: string;
  text: string;
  sources: Source[];
  snippets: CodeSnippet[];
  unavailable: string[]; // sources that were missing or failed; never papered over
}

export type ResponseCardStatus = "pending" | "spoken" | "sent_to_chat" | "dismissed";
export type ResponseActionName = "speak" | "send_to_chat" | "dismiss" | "show_on_stage";

/** Shared answer to a voice question. Silent until a participant chooses Speak. */
export interface ResponseCard {
  id: string;
  meeting_id: string;
  invocation: Invocation;
  answer: Answer;
  status: ResponseCardStatus;
  urgency: HandUrgency;
}

export interface ResponseAction {
  card_id: string;
  action: ResponseActionName;
  by_id: string;
}

export interface QuestionAnswered {
  id: string;
  asked_by: string;
  question: string;
  answer: string;
  snippet_id?: string | null;
  via: InvocationVia;
  t?: number | null;
}

/** A claim checked against the team's records during a live meeting. It is never shown to the
 * room: the agent sends it as a private chat message to recipient_id, the participant who made the
 * claim. The copy kept for the write-up has no recipient_id. */
export interface FactCheck {
  id: string;
  claim: string;
  speaker_name: string;
  verdict: Verdict;
  confidence: number;
  severity: Severity;
  finding: string; // what the records show, in one short sentence
  snippet_ids: string[];
  sources: Source[];
  recipient_id?: string | null; // the claimant's participant id
  t?: number | null;
  created_at?: string | null;
}
