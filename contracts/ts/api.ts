// HTTP request and response bodies. Board -> brain, and realtime -> brain (internal).
import type { Agenda, AgendaItemStatus, AgendaNudge } from "./agenda";
import type { Answer, CodeSnippet, FactCheck, Invocation, Visibility } from "./agent";
import type { Meeting, Person } from "./meeting";
import type { TranscriptSegment } from "./transcript";

export type AskTurnRole = "user" | "agent";

/** POST /auth/login. There is no public sign-up; accounts come from `brain add-user` or an admin (POST /team/accounts). */
export interface LoginRequest {
  email: string;
  password: string;
}

/** The board sends `token` as `Authorization: Bearer <token>` until `expires_at`. */
export interface LoginResponse {
  token: string;
  expires_at: string; // ISO 8601
  person: Person;
}

/** POST /auth/password while signed in. The new password is at least 10 characters. */
export interface PasswordChange {
  current_password: string;
  new_password: string;
}

/** POST /team/accounts, admin only: a person on the admin's own team with an email login. */
export interface CreateAccountRequest {
  name: string;
  email: string;
  title?: string | null;
  is_admin?: boolean;
}

/** The generated password is returned only this once; nothing is emailed. */
export interface CreateAccountResponse {
  person: Person;
  password: string;
}

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

/** The worker sends each check to its recipient_id only, as a private chat message from the agent
 * (Topic.PRIVATE_CHAT), never stored, spoken or shown to the room. `snippets` are the code the
 * checks' snippet_ids name. */
export interface FactCheckResponse {
  checks?: FactCheck[];
  snippets?: CodeSnippet[];
}

/** realtime -> brain when someone joins a live meeting 5 minutes or more after it started, or
 * comes back after 5 minutes or more away. `since` and `until` are seconds from the meeting start:
 * the span they missed. */
export interface CatchUpRequest {
  participant_id: string;
  since: number;
  until: number;
}

/** What the worker sends only to the participant, as a private chat message from the agent
 * (Topic.PRIVATE_CHAT, recipient_id set). `text` is null when there is nothing to send.
 * `source_times` are the meeting seconds of the transcript lines it rests on. Never stored,
 * broadcast or spoken. */
export interface CatchUpResponse {
  text?: string | null;
  source_times?: number[];
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

/** Words the worker's Scribe streams are biased toward, most important first, already within
 * Scribe Realtime's limits (brain.keyterms). */
export interface KeytermsResponse {
  terms: string[];
}

/** realtime -> brain before the worker acts on a shared answer card (Speak, Post in chat,
 * Dismiss) for a participant: whether TeamSettings.who_can_allow lets them. With "host", only the
 * meeting's host or an admin; with "everyone", any participant. Never the agent itself. */
export interface CardPermissionResponse {
  allowed: boolean;
}

/** realtime -> brain when the worker joins a meeting's room (named by the meeting id) and before
 * it speaks. Segment times are seconds from meeting.started_at. voice_id is the team's chosen
 * agent voice; null means the worker's default (ELEVENLABS_VOICE_ID). */
export interface WorkerMeetingResponse {
  meeting: Meeting;
  voice_id?: string | null;
}
