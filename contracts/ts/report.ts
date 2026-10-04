import type { CodeSnippet, FactCheck, QuestionAnswered, Severity, Source } from "./agent";

export type DecisionStatus = "active" | "superseded";
export type DecisionRelationType = "contradicts" | "superseded_by";
export type JiraStatus = "draft" | "todo" | "in_progress" | "done";
export type TaskDestination = "jira" | "github";

export interface DecisionRelation {
  type: DecisionRelationType;
  decision_id: string;
}

/** One thing said on the way to a decision: a few words on what, and where it was said. */
export interface DecisionStep {
  text: string;
  t: number; // seconds from the meeting start, where this part of the conversation begins
  seg_ids: string[]; // the transcript segments it describes, in time order
}

export interface Decision {
  id: string;
  meeting_id: string;
  text: string;
  made_by: string;
  t: number;
  quote: string;
  /** What was said that led to it, oldest first. Empty when none was recorded: the decision then
   * has only its quote. */
  chain?: DecisionStep[];
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
  url?: string | null; // where the pushed issue opens, when the site's address was known
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
  updated_at?: string | null; // ISO datetime the write-up last saved progress
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
  warning?: string | null;
}
