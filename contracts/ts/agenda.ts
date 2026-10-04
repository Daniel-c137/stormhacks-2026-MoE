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
}

export interface Agenda {
  meeting_id: string;
  items: AgendaItem[];
  generated_at: string; // ISO 8601
  updated_at?: string | null; // ISO 8601; last human edit
  current_item_id?: string | null; // being discussed now; null when off the agenda
  tracked_until?: number | null; // transcript seconds tracked so far; null before any
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
  item_id: string;
  text: string;
}
