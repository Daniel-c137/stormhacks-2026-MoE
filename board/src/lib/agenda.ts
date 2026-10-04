import type { AgendaItem } from "@moe/contracts";
import { getAgenda, suggestAgenda, updateAgenda } from "./api";

/** A title as compared for duplicates: without case, punctuation or extra spaces (as the brain does). */
export const agendaTitleKey = (title: string) =>
  title
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();

/** The drafted items to add after `current`: none repeating a title already there or earlier in the draft. */
export function draftedTopics(current: { title: string }[], drafted: AgendaItem[]): AgendaItem[] {
  const seen = new Set(current.map((t) => agendaTitleKey(t.title)));
  return drafted.filter((item) => {
    const key = agendaTitleKey(item.title);
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/** Asks the agent for a draft and adds it after the meeting's saved items, as ordinary items. Returns how
 * many were added. Used when scheduling; the lobby does the same with its own edit queue. */
export async function draftAgendaInto(meetingId: string): Promise<number> {
  const current = await getAgenda(meetingId);
  const result = await suggestAgenda(meetingId);
  const added = draftedTopics(current.items, result.items);
  if (added.length) {
    const keep = current.items.map((i) => ({ id: i.id, title: i.title, minutes: i.minutes ?? null }));
    const add = added.map((i) => ({ title: i.title, minutes: i.minutes ?? null }));
    await updateAgenda(meetingId, { items: [...keep, ...add] });
  }
  return added.length;
}
