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
}

export interface Agenda {
  meeting_id: string;
  items: AgendaItem[];
  generated_at: string; // ISO 8601
  updated_at?: string | null; // ISO 8601; last human edit
}

/** A reminder that an agenda item has not come up yet. */
export interface AgendaNudge {
  meeting_id: string;
  item_id: string;
  text: string;
}
