"use client";

import { type TaskDraft, identity } from "@moe/contracts";
import Link from "next/link";
import { useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Icon, type IconName } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";
import { useDecisions, useMeetings, useSettings, useTasks } from "@/hooks/useApi";
import { fmtDate, fmtT, isoDay, meetingStart } from "@/lib/format";

const JIRA_ICON: Record<TaskDraft["jira_status"], IconName> = {
  draft: "circle-dashed",
  todo: "circle",
  in_progress: "circle-dot",
  done: "circle-check",
};
const JIRA_LABEL: Record<TaskDraft["jira_status"], string> = {
  draft: "Not pushed",
  todo: "To do",
  in_progress: "In progress",
  done: "Done",
};

/** Searchable decisions across meetings (superseded and contradicting links) and every task. */
export function DecisionsAndTasks() {
  const { me, members, person } = useTeam();
  const decisions = useDecisions();
  const tasks = useTasks();
  const meetings = useMeetings();
  const settings = useSettings();
  const [tab, setTab] = useState<"decisions" | "tasks">("decisions");
  const [query, setQuery] = useState("");
  const [owner, setOwner] = useState(me.id);

  const meetingOf = (id: string) => meetings.data?.find((m) => m.id === id);
  const where = (meetingId: string) => {
    const m = meetingOf(meetingId);
    const start = m ? meetingStart(m) : null;
    return [m?.title ?? "A meeting", start && fmtDate(start, true)].filter(Boolean).join(" · ");
  };

  const all = decisions.data ?? [];
  const q = query.trim().toLowerCase();
  const shown = all.filter(
    (d) => !q || [d.text, d.quote, d.made_by, meetingOf(d.meeting_id)?.title ?? ""].join(" ").toLowerCase().includes(q),
  );
  const rows = (tasks.data ?? []).filter((t) => owner === "all" || t.owner_id === owner);
  const today = isoDay(new Date());
  const site = settings.data?.jira.site;

  return (
    <main className="rep-wrap">
      <Link className="back-link" href="/">
        <Icon name="arrow-left" />
        Meetings
      </Link>
      <div className="mem-head">
        <h1>Decisions &amp; tasks</h1>
        <div className="seg" role="radiogroup" aria-label="Show">
          <button type="button" role="radio" aria-checked={tab === "decisions"} onClick={() => setTab("decisions")}>
            Decisions{decisions.data ? ` ${all.length}` : ""}
          </button>
          <button type="button" role="radio" aria-checked={tab === "tasks"} onClick={() => setTab("tasks")}>
            Tasks{tasks.data ? ` ${tasks.data.length}` : ""}
          </button>
        </div>
      </div>

      {tab === "decisions" ? (
        <section className="rcard" aria-label="Decisions">
          <p className="muted-p" style={{ marginBottom: 16 }}>
            Everything {identity.agent_name} has recorded across your meetings. Decisions that were later reversed stay
            here, marked as superseded.
          </p>
          <div className="t-tools">
            <label className="t-search">
              <Icon name="search" />
              <input
                className="field"
                type="search"
                aria-label="Search decisions"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search decisions"
              />
            </label>
            {decisions.data && (
              <span className="t-count">
                {shown.length} of {all.length}
              </span>
            )}
          </div>
          {decisions.error && !decisions.data && (
            <Notice error={decisions.error} onRetry={decisions.reload}>
              Decisions can&apos;t be loaded.
            </Notice>
          )}
          {decisions.loading && !decisions.data && <p className="muted-p">Loading…</p>}
          {decisions.data && !shown.length && <p className="muted-p">{all.length ? "No decisions match." : "No decisions recorded yet."}</p>}
          {shown.map((d, i) => {
            const superseded = d.status === "superseded";
            const related = d.relation ? all.find((x) => x.id === d.relation?.decision_id) : undefined;
            return (
              <div key={d.id} className="decision" data-superseded={superseded}>
                <span className="dnum">{String(i + 1).padStart(2, "0")}</span>
                <div>
                  <p className="dtext">{d.text}</p>
                  {d.quote && <blockquote className="dquote">“{d.quote}”</blockquote>}
                  <div className="dmeta">
                    <Icon name={superseded ? "archive" : related ? "git-compare" : "circle-check"} />
                    <span>{superseded ? "Superseded" : related ? "Active · reverses an earlier decision" : "Active"}</span>
                    <span>
                      {d.made_by} · {fmtT(d.t)}
                    </span>
                    <Link href={`/meetings/${d.meeting_id}`}>{where(d.meeting_id)}</Link>
                  </div>
                  {related && (
                    <div className="contra" role="note">
                      <Icon name={superseded ? "arrow-right" : "undo-2"} />
                      <div style={{ minWidth: 0 }}>
                        <div className="contra-h">{superseded ? "Superseded by" : "Reverses"}</div>
                        <p>“{related.text}”</p>
                        <Link className="link" href={`/meetings/${related.meeting_id}`}>
                          {where(related.meeting_id)}
                        </Link>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </section>
      ) : (
        <section className="rcard" aria-label="Tasks">
          <div className="t-tools">
            <div className="chips" role="group" aria-label="Filter by owner">
              {[["all", "Everyone"], [me.id, "You"], ...members.filter((p) => p.id !== me.id).map((p) => [p.id, p.short])].map(
                ([id, label]) => (
                  <button key={id} type="button" className="chip" onClick={() => setOwner(id)} aria-pressed={owner === id}>
                    {label}
                  </button>
                ),
              )}
            </div>
            {tasks.data && (
              <span className="t-count">
                {rows.length} tasks · {rows.filter((t) => t.jira_status !== "done").length} open
              </span>
            )}
          </div>
          {tasks.error && !tasks.data && (
            <Notice error={tasks.error} onRetry={tasks.reload}>
              Tasks can&apos;t be loaded.
            </Notice>
          )}
          {tasks.loading && !tasks.data && <p className="muted-p">Loading…</p>}
          {tasks.data && !rows.length && <p className="muted-p">No tasks here.</p>}
          {rows.map((t) => {
            const overdue = Boolean(t.due && t.due < today && t.jira_status !== "done");
            return (
              <div key={t.id} className="trow" data-status={t.jira_status}>
                <Icon name={JIRA_ICON[t.jira_status]} />
                <div style={{ minWidth: 0 }}>
                  <div className="trow-title">{t.title}</div>
                  <div className="trow-meta">
                    <span>{JIRA_LABEL[t.jira_status]}</span>
                    {t.owner_id && <span>{person(t.owner_id).short}</span>}
                    {t.due && <span className={overdue ? "overdue" : undefined}>due {fmtDate(new Date(`${t.due}T12:00:00`))}</span>}
                    <Link href={`/meetings/${t.meeting_id}`}>{meetingOf(t.meeting_id)?.title ?? "Meeting"}</Link>
                  </div>
                </div>
                {t.key &&
                  (site ? (
                    <a className="jump" href={`https://${site}/browse/${t.key}`} target="_blank" rel="noopener noreferrer">
                      {t.key}
                    </a>
                  ) : (
                    <span className="jump">{t.key}</span>
                  ))}
              </div>
            );
          })}
        </section>
      )}
    </main>
  );
}
