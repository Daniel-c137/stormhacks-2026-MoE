import type { Meeting, Person } from "@moe/contracts";

export const cx = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(" ");

const pad = (n: number) => String(n).padStart(2, "0");

/** Seconds from the meeting start as mm:ss. */
export function fmtT(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${pad(Math.floor(s / 60))}:${pad(s % 60)}`;
}

export const fmtClock = (d: Date) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;

export const fmtDate = (d: Date, withYear = false) =>
  d.toLocaleDateString("en-US", { month: "short", day: "numeric", ...(withYear ? { year: "numeric" } : {}) });

export const fmtLongDate = (d: Date) =>
  d.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });

export const isSameDay = (a: Date, b: Date) => a.toDateString() === b.toDateString();

/** yyyy-mm-dd in local time, for date inputs. */
export const isoDay = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

export const fmtDuration = (min: number) =>
  min < 60 ? `${min} min` : `${min / 60} ${min === 60 ? "hour" : "hours"}`;

export function initialsOf(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  return (words.map((w) => w[0]).join("").slice(0, 2) || "?").toUpperCase();
}

export const shortOf = (name: string) => name.trim().split(/\s+/)[0] || name;

/** When a meeting starts: its scheduled time until it has actually started. */
export function meetingStart(m: Meeting): Date | null {
  const iso = m.started_at ?? m.scheduled_start;
  return iso ? new Date(iso) : null;
}

/** "14:00–now", "14:00–14:47" or just "14:00"; prefixed with the date when it is not today. */
export function meetingTimeText(m: Meeting, now: Date): string {
  const start = meetingStart(m);
  if (!start) return "";
  const day = isSameDay(start, now) ? "" : `${fmtDate(start)} · `;
  if (m.status === "live") return `${day}${fmtClock(start)}–now`;
  if (!m.duration_min) return `${day}${fmtClock(start)}`;
  return `${day}${fmtClock(start)}–${fmtClock(new Date(start.getTime() + m.duration_min * 60_000))}`;
}

export const STATUS_LABEL: Record<Meeting["status"], string> = {
  scheduled: "Scheduled",
  live: "Live",
  processing: "Processing",
  needs_review: "Needs review",
  pushed: "Pushed",
};

export function joinNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** A stand-in for an account the members list does not include. */
export const unknownPerson = (id: string): Person => ({ id, name: "Unknown", short: "Unknown", initials: "?" });

/** Pull a meeting code out of a pasted link, or take the text as the code. */
export function parseMeetingCode(input: string): string {
  const text = input.trim();
  const fromLink = /\/m\/([^/?#\s]+)/.exec(text);
  return fromLink ? decodeURIComponent(fromLink[1]) : text;
}
