import type { CodeSnippet, FactCheck, QuestionAnswered, Severity, Source } from "./agent";

export type DecisionStatus = "active" | "superseded";
export type DecisionRelationType = "contradicts" | "superseded_by";
export type JiraStatus = "draft" | "todo" | "in_progress" | "done";
export type TaskDestination = "jira" | "github";

export interface DecisionRelation {
  type: DecisionRelationType;
  decision_id: string;
}

export interface Decision {
  id: string;
  meeting_id: string;
  text: string;
  made_by: string;
  t: number;
  quote: string;
  status: DecisionStatus;
  relation?: DecisionRelation | null;
}

/** Internal task draft. Owner and due date only when stated or clearly inferable. */
export interface TaskDraft {
  id: string;
  meeting_id: string;
  title: string;
  description?: string | null;
  owner_id?: string | null;
  due?: string | null; // YYYY-MM-DD
  t?: number | null;
  quote?: string | null;
  include: boolean;
  key?: string | null; // external key once pushed
  jira_status: JiraStatus;
}

export interface Risk {
  text: string;
  severity: Severity;
}

/** Post-meeting write-up. Empty sections stay empty; nothing is invented to fill them. */
export interface Report {
  meeting_id: string;
  summary: string;
  topics: string[];
  decisions: Decision[];
  tasks: TaskDraft[];
  risks: Risk[];
  open_questions: string[];
  blockers: string[];
  technical_context: string[];
  links: Source[];
  fact_checks: FactCheck[];
  questions: QuestionAnswered[];
  snippets: CodeSnippet[];
}

export interface ReportProgress {
  meeting_id: string;
  steps: string[];
  current: number;
  done: boolean;
  error?: string | null; // why the pipeline stopped at the current step
}

/** A specific human approval: which drafts go where. */
export interface TaskPushRequest {
  task_ids: string[];
  destination: TaskDestination;
  approved_by: string;
}

export interface TaskPushResult {
  task_id: string;
  key?: string | null;
  url?: string | null;
  error?: string | null;
}
