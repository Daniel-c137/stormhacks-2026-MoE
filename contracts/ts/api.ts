// HTTP request and response bodies. Board -> brain, and realtime -> brain (internal).
import type { AgendaItem } from "./agenda";
import type { Answer, Invocation, Visibility } from "./agent";
import type { Meeting } from "./meeting";
import type { TranscriptSegment } from "./transcript";

/** No scheduled_for starts the meeting now. Invitees are team members; the creator organises. */
export interface CreateMeetingRequest {
  title: string;
  scheduled_for?: string | null; // ISO 8601
  duration_min?: number | null;
  invitee_ids?: string[];
}

/** The signed-in user's own profile. photo is an image data URL; null removes it. */
export interface UpdateProfileRequest {
  name: string;
  photo?: string | null;
}

/** Topics people add in the lobby replace the meeting's agenda items. */
export interface UpdateAgendaRequest {
  items: AgendaItem[];
}

/** Tidy one agenda topic. The text comes back for the user to accept; nothing is saved. */
export interface RewriteTopicRequest {
  text: string;
}

export interface RewriteTopicResponse {
  text: string;
}

export interface JoinMeetingResponse {
  meeting: Meeting;
  livekit_url: string;
  token: string;
}

/** A typed question from the board. Private answers go back only to the asker. */
export interface AskRequest {
  question: string;
  visibility: Visibility;
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
