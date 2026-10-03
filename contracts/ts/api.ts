// HTTP request and response bodies. Board -> brain, and realtime -> brain (internal).
import type { Answer, Invocation, Visibility } from "./agent";
import type { Meeting } from "./meeting";
import type { TranscriptSegment } from "./transcript";

export interface CreateMeetingRequest {
  title: string;
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
