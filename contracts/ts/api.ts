// HTTP request and response bodies. Board -> brain, and realtime -> brain (internal).
import type { Agenda, AgendaItemStatus, AgendaNudge } from "./agenda";
import type { AgentState, Answer, CodeSnippet, FactCheck, Invocation, Visibility } from "./agent";
import type { Meeting } from "./meeting";
import type { TranscriptSegment } from "./transcript";

export type AskTurnRole = "user" | "agent";

/** Without scheduled_start the meeting starts now; with it, the meeting waits as scheduled. */
export interface CreateMeetingRequest {
  title: string;
  scheduled_start?: string | null; // ISO 8601
  duration_min?: number | null;
  invitee_ids?: string[];
}

export interface JoinMeetingResponse {
  meeting: Meeting;
  livekit_url: string;
  token: string;
}

export interface InviteRequest {
  person_ids: string[];
}

/** An earlier turn of the same conversation, sent back so a follow-up has its context. */
export interface AskTurn {
  role: AskTurnRole;
  text: string;
}

/** A typed question from the board. Private answers go back only to the asker. */
export interface AskRequest {
  question: string;
  visibility: Visibility;
  history?: AskTurn[]; // oldest first
}

/** An agenda item as a person edits it. No id means a new item. */
export interface AgendaItemInput {
  id?: string | null;
  title: string;
  minutes?: number | null;
  status?: AgendaItemStatus | null; // null keeps the item's status; new items are pending
}

/** The whole edited list, in order. Items left out are removed. */
export interface AgendaUpdate {
  items: AgendaItemInput[];
}

export interface AgendaRewriteRequest {
  text: string;
}

export interface AgendaRewriteResponse {
  text: string;
}

/** realtime -> brain on a timer. `now` is seconds from the meeting start; omitted, the brain
 * takes it from started_at. A finite time no later than the real time since the start (plus a
 * minute of slack). */
export interface AgendaTrackRequest {
  now?: number | null;
}

/** The worker publishes `agenda` on Topic.AGENDA and each nudge on Topic.AGENDA_NUDGE. */
export interface AgendaTrackResponse {
  agenda: Agenda;
  nudges?: AgendaNudge[];
}

/** realtime -> brain on a timer, never per utterance. `now` is seconds from the meeting start;
 * omitted, the brain takes it from started_at. */
export interface FactCheckRequest {
  now?: number | null;
}

/** The worker publishes each check on Topic.FACT_CHECK: a public one to the room, a private one
 * only to its recipient_id. `agent_state`, set only when a check raises the hand, goes on
 * Topic.AGENT_STATE; the hand is a visual cue and nothing is spoken. `snippets` are the code the
 * checks' snippet_ids name, for whoever sees those checks. */
export interface FactCheckResponse {
  checks?: FactCheck[];
  agent_state?: AgentState | null;
  snippets?: CodeSnippet[];
}

/** Only the fields that are set change. */
export interface ProfileUpdate {
  name?: string | null;
}

export interface SegmentsIngest {
  segments: TranscriptSegment[];
}

export interface InvokeRequest {
  invocation: Invocation;
  recent_segments: TranscriptSegment[];
}

export interface InvokeResponse {
  answer: Answer;
}
