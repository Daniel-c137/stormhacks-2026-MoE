"use client";

import type { TaskDraft } from "@moe/contracts";
import Link from "next/link";
import { useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Icon, type IconName } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";
import { useMeetings, useSettings, useTasks } from "@/hooks/useApi";
import { fmtDate, isoDay } from "@/lib/format";
import { jiraIssueUrl } from "@/lib/links";

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

/** Every task from the team's meetings, by owner: yours first. */
export function TaskList() {
  const { me, members, person } = useTeam();
  const tasks = useTasks();
  const meetings = useMeetings();
  const settings = useSettings();
  const [owner, setOwner] = useState(me.id);

  const meetingOf = (id: string) => meetings.data?.find((m) => m.id === id);
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
        <h1>Tasks</h1>
      </div>

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
                  <a className="jump" href={jiraIssueUrl(site, t.key) ?? undefined} target="_blank" rel="noopener noreferrer">
                    {t.key}
                  </a>
                ) : (
                  <span className="jump">{t.key}</span>
                ))}
            </div>
          );
        })}
      </section>
    </main>
  );
}
