"use client";

import { AGENT_PARTICIPANT_ID, type Agenda, type AgendaItem, identity } from "@moe/contracts";
import { useCallback, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";
import { useDismiss } from "@/hooks/useDismiss";
import { fmtClock } from "@/lib/format";

export interface AgendaTrackerProps {
  agenda: Agenda | null;
  /** Why the agenda is missing, if it is. */
  error: Error | null;
  meId: string;
  /** When the meeting started, to turn an item's covered_t into a time of day. */
  startedAt: string | null | undefined;
  /** A teammate's short name by id; null for someone the team list does not know. */
  nameOf: (id: string) => string | null;
  /** A person ticks an item as covered, or unticks it. */
  onCheck: (item: AgendaItem, covered: boolean) => void;
}

const RING = 2 * Math.PI * 8;

/** Progress as a ring that fills up; a solid tick once everything is covered. */
function Ring({ done, total }: { done: number; total: number }) {
  if (total > 0 && done === total) {
    return (
      <span className="at-ring" data-done="true" aria-hidden="true">
        <Icon name="check" />
      </span>
    );
  }
  return (
    <svg className="at-ring" viewBox="0 0 22 22" aria-hidden="true">
      <circle cx="11" cy="11" r="8" className="track" />
      {done > 0 && <circle cx="11" cy="11" r="8" className="fill" strokeDasharray={`${(done / total) * RING} ${RING}`} />}
    </svg>
  );
}

/** "3 min", "45 s": talk time so far, rounded down so it never runs ahead of the tracker. */
function used(seconds: number): string {
  return seconds < 60 ? `${Math.floor(seconds)} s` : `${Math.floor(seconds / 60)} min`;
}

/** Talk time against the timebox, when the tracker has attributed any: "4 min / 10 min". */
function timeText(item: AgendaItem): string | null {
  const spent = item.discussed_s ?? 0;
  if (!item.minutes) return spent > 0 ? used(spent) : null;
  const over = spent - item.minutes * 60;
  if (over > 0) return `${used(over)} over ${item.minutes} min`;
  return spent > 0 ? `${used(spent)} / ${item.minutes} min` : null;
}

/** The agenda next to the meeting title: how much is covered, and the list behind it. The agent
 * ticks items off as it keeps time; anyone can tick or untick one themselves. */
export function AgendaTracker({ agenda, error, meId, startedAt, nameOf, onCheck }: AgendaTrackerProps) {
  const agent = identity.agent_name;
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  useDismiss(
    wrap,
    open,
    useCallback(() => setOpen(false), []),
  );

  const items = agenda?.items ?? [];
  const total = items.length;
  const done = items.filter((i) => i.status === "covered").length;
  const summary = !total ? "Nothing on the agenda" : done === total ? "Everything on the agenda is covered" : `${done} of ${total} covered`;

  const meta = (item: AgendaItem): string => {
    if (item.status === "covered") {
      const at =
        startedAt && item.covered_t != null ? fmtClock(new Date(Date.parse(startedAt) + item.covered_t * 1000)) : null;
      const who =
        item.covered_by === AGENT_PARTICIPANT_ID
          ? `Covered · checked by ${agent}`
          : item.covered_by === meId
            ? "Checked by you"
            : item.covered_by
              ? `Checked by ${nameOf(item.covered_by) ?? "a teammate"}`
              : "Covered";
      return at ? `${who} · ${at}` : who;
    }
    const state = item.status === "skipped" ? "Skipped" : item.id === agenda?.current_item_id ? "Being discussed now" : "Not covered yet";
    const time = timeText(item);
    return time ? `${state} · ${time}` : state;
  };

  return (
    <div className="at-wrap" ref={wrap}>
      <button
        type="button"
        className="fx-pill at-pill"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={total ? `Agenda: ${done} of ${total} covered` : "Agenda"}
      >
        <Ring done={done} total={total} />
        <span className="at-label">Agenda</span>
        {total > 0 && (
          <span className="at-count">
            {done}/{total}
          </span>
        )}
        <Icon name={open ? "chevron-up" : "chevron-down"} />
      </button>
      {open && (
        <div className="at-pop" role="dialog" aria-label="Agenda">
          <div className="at-head">
            <div>
              <h2>Agenda</h2>
              <p aria-live="polite">{agenda ? summary : error ? "" : "Loading the agenda…"}</p>
            </div>
            <button type="button" className="icon-btn" onClick={() => setOpen(false)} aria-label="Close the agenda">
              <Icon name="x" />
            </button>
          </div>
          {!agenda && error && <Notice error={error}>The agenda couldn&apos;t be loaded.</Notice>}
          {total > 0 && (
            <>
              <div className="at-bar" aria-hidden="true">
                <span style={{ width: `${(done / total) * 100}%` }} />
              </div>
              <ul className="at-list">
                {items.map((item) => {
                  const covered = item.status === "covered";
                  return (
                    <li key={item.id} className="at-item" data-status={item.status} data-current={item.id === agenda?.current_item_id}>
                      <button
                        type="button"
                        className="at-check"
                        role="checkbox"
                        aria-checked={covered}
                        aria-label={covered ? `Mark “${item.title}” as not covered` : `Mark “${item.title}” as covered`}
                        onClick={() => onCheck(item, !covered)}
                      >
                        {covered && <Icon name="check" />}
                      </button>
                      <div className="at-main">
                        <span className="at-title">{item.title}</span>
                        <span className="at-meta">{meta(item)}</span>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
