import type { Source } from "./agent";

export type AgendaItemStatus = "pending" | "covered" | "skipped";

export interface AgendaItem {
  id: string;
  title: string;
  why?: string | null;
  owner_id?: string | null;
  sources: Source[];
  status: AgendaItemStatus;
}

export interface Agenda {
  meeting_id: string;
  items: AgendaItem[];
  generated_at: string; // ISO 8601
}

/** A reminder that an agenda item has not come up yet. */
export interface AgendaNudge {
  meeting_id: string;
  item_id: string;
  text: string;
}
