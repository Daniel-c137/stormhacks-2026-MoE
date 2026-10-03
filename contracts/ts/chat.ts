import type { Visibility } from "./agent";

/** Public messages are saved with the meeting. Private ones are ephemeral and never stored. */
export interface ChatMessage {
  id: string;
  meeting_id: string;
  sender_id: string;
  sender_name: string;
  is_agent: boolean;
  text: string;
  ts: string; // ISO 8601
  visibility: Visibility;
  recipient_id?: string | null;
  snippet_id?: string | null;
}
