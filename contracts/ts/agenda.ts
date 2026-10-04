import type { Source } from "./agent";

export type AgendaItemStatus = "pending" | "covered" | "skipped";

export interface AgendaItem {
  id: string;
  title: string;
  why?: string | null;
  owner_id?: string | null;
  sources: Source[];
  status: AgendaItemStatus;
  minutes?: number | null; // timebox
  added_by?: string | null; // person id; null when the agent proposed it
  // Timekeeping during the meeting; times are seconds from the meeting start.
  discussed_s?: number; // talk time attributed to this item so far
  nudged_t?: number | null; // when the agent nudged that it had not come up; once at most
  // The end of the last thing said about it, as the tracker labelled it. Null until it comes
  // up, and again once a person reopens it: the tracker only covers an item that has come up
  // since.
  last_discussed_t?: number | null;
  // Who marked it covered: a person id, or AGENT_PARTICIPANT_ID when the tracker did. Null
  // while it is not covered. covered_t is when: for the tracker, when the item's discussion
  // ended (not when it noticed); for a person, when they ticked it. null if the meeting had
  // not started.
  covered_by?: string | null;
  covered_t?: number | null;
}

/** One person's agenda for a meeting: everyone has their own, and nobody else sees or edits it. */
export interface Agenda {
  meeting_id: string;
  person_id?: string | null; // whose agenda; always set once saved
  items: AgendaItem[];
  generated_at: string; // ISO 8601
  updated_at?: string | null; // ISO 8601; last human edit
  current_item_id?: string | null; // being discussed now; null when off the agenda
  tracked_until?: number | null; // end of the last caption tracked; null before any
  revision?: number; // bumped by every save; 0 until first saved
}

/** Proposed items, not saved; a person adds the ones they want. */
export interface AgendaSuggestions {
  items: AgendaItem[];
  unavailable: string[]; // sources that were missing or failed; never papered over
}

/** A reminder that an agenda item has not come up yet. */
export interface AgendaNudge {
  meeting_id: string;
  person_id?: string | null; // whose agenda the item is on; the nudge goes only to them
  item_id: string;
  text: string;
}
