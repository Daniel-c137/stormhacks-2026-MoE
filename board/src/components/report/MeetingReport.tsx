"use client";

import {
  AGENT_PARTICIPANT_ID,
  type Decision,
  type DecisionStep,
  type Meeting,
  type Report,
  type TaskDestination,
  type TaskDraft,
  type TeamSettings,
  type TranscriptSegment,
  identity,
} from "@moe/contracts";
import Link from "next/link";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Icon, type IconName, Spinner } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { Notice } from "@/components/ui/Notice";
import { Sources } from "@/components/ui/Sources";
import {
  useDecisions,
  useMeeting,
  useMeetings,
  useReport,
  useReportProgress,
  useSavedTranscript,
  useSettings,
} from "@/hooks/useApi";
import { describeError, pushTasks, retryReport, updateTask } from "@/lib/api";
import { attendeeNames, fmtClock, fmtDate, fmtLongDate, fmtT, joinNames, meetingStart, shortOf } from "@/lib/format";
import { hostOrAdmin } from "@/lib/roles";
import { jiraIssueUrl } from "@/lib/links";
import { ListenButton } from "./ListenButton";
import { TaskReview } from "./TaskReview";

export interface MeetingReportProps {
  meetingId: string;
}

const BackLink = () => (
  <Link className="back-link" href="/">
    <Icon name="arrow-left" />
    Meetings
  </Link>
);

/**
 * Processing steps while the report is written, then summary, decisions (each with what was said
 * that led to it, and links to past decisions), task review, and whatever else the meeting
 * produced, down to the transcript.
 */
export function MeetingReport({ meetingId }: MeetingReportProps) {
  // Poll while the meeting is still running or being written up.
  const [watching, setWatching] = useState(true);
  const meeting = useMeeting(meetingId, watching ? 4000 : 0);
  const status = meeting.data?.status;
  useEffect(() => {
    if (status) setWatching(status === "scheduled" || status === "live" || status === "processing");
  }, [status]);

  if (!meeting.data) {
    return (
      <main className="rep-wrap">
        <BackLink />
        <section className="proc">
          {meeting.error ? (
            <Notice error={meeting.error} onRetry={meeting.reload}>
              This meeting can&apos;t be loaded.
            </Notice>
          ) : (
            <p className="muted-p">Loading…</p>
          )}
        </section>
      </main>
    );
  }
  const m = meeting.data;
  if (m.status === "scheduled" || m.status === "live") {
    return (
      <main className="rep-wrap">
        <section className="proc">
          <div>
            <p className="eyebrow">{m.title}</p>
            <h1>{m.status === "live" ? "This meeting is still going" : "This meeting hasn't started"}</h1>
            <p className="proc-note">The report is written once the meeting ends.</p>
          </div>
          <Link className="btn btn-primary" href={`/m/${m.code}`}>
            {m.status === "live" ? "Join" : "Open the lobby"} <Icon name="arrow-right" />
          </Link>
        </section>
      </main>
    );
  }
  return (
    <main className="rep-wrap">
      {m.status === "processing" ? <Processing meeting={m} /> : <ReportView meeting={m} onPushed={meeting.reload} />}
    </main>
  );
}

function Processing({ meeting }: { meeting: Meeting }) {
  const agent = identity.agent_name;
  const { me } = useTeam();
  const progress = useReportProgress(meeting.id, 2000);
  const [retrying, setRetrying] = useState(false);
  const [retryProblem, setRetryProblem] = useState("");
  const stopped = progress.data?.error;
  const retry = async () => {
    setRetrying(true);
    setRetryProblem("");
    try {
      await retryReport(meeting.id);
      progress.reload();
    } catch (err) {
      setRetryProblem(`The write-up didn't restart. ${describeError(err)}`);
    } finally {
      setRetrying(false);
    }
  };
  const steps = progress.data?.steps ?? [];
  const current = progress.data?.current ?? 0;
  const meta = [meeting.duration_min ? `${meeting.duration_min} min` : null, `${meeting.participant_ids.length} people`]
    .filter(Boolean)
    .join(" · ");
  return (
    <section className="proc" aria-live="polite">
      <span className="mark-slot">
        <Mark size={64} tile state="working" />
      </span>
      <div>
        <p className="eyebrow">
          {meeting.title} · {meta}
        </p>
        <h1>{agent} is writing up the meeting</h1>
        <p className="proc-note">
          Reports are usually ready a few minutes after the meeting ends. You can leave this page; it will show up under
          Meetings as Needs review. Nothing goes to Jira until you have reviewed it.
        </p>
      </div>
      {steps.length > 0 && (
        <ol className="steps">
          {steps.map((step, i) => {
            const state = progress.data?.done || i < current ? "Done" : i === current ? "In progress" : "";
            return (
              <li key={step} className="step" data-state={state}>
                <span className="ic">
                  {state === "In progress" ? <Spinner size={18} /> : <Icon name={state === "Done" ? "circle-check" : "circle"} />}
                </span>
                <span>{step}</span>
                <span className="st">{state}</span>
              </li>
            );
          })}
        </ol>
      )}
      {progress.error && !progress.data && <Notice error={progress.error}>Progress isn&apos;t available.</Notice>}
      {stopped && (
        <Notice>
          The write-up stopped: {stopped}
          {hostOrAdmin(meeting, me) ? "" : " The host or an admin can start it again."}
        </Notice>
      )}
      {stopped && hostOrAdmin(meeting, me) && (
        <button type="button" className="btn btn-primary" onClick={() => void retry()} disabled={retrying}>
          {retrying ? <Spinner /> : <Icon name="rotate-ccw" />}
          Try the write-up again
        </button>
      )}
      {retryProblem && (
        <span role="alert" className="err">
          {retryProblem}
        </span>
      )}
      <Link className="btn btn-outline" href="/">
        <Icon name="arrow-left" />
        Back to meetings
      </Link>
    </section>
  );
}

function Bullets({ id, title, icon, items }: { id: string; title: string; icon: IconName; items: ReactNode[] }) {
  if (!items.length) return null;
  return (
    <section className="rcard" aria-labelledby={id}>
      <h2 id={id} className="sec-h">
        {title}
      </h2>
      <ul className="bullets">
        {items.map((item, i) => (
          <li key={i}>
            <Icon name={icon} />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Lines shown for one step before it falls back to only the lines the step names. */
const TALK_MAX = 8;

/** The part of the conversation a step describes: the lines it names and those between them. */
function talkOf(step: DecisionStep, segs: TranscriptSegment[]): TranscriptSegment[] {
  const at = step.seg_ids.map((id) => segs.findIndex((g) => g.seg_id === id)).filter((i) => i >= 0);
  if (!at.length) return [];
  const first = Math.min(...at);
  const last = Math.max(...at);
  return last - first < TALK_MAX ? segs.slice(first, last + 1) : at.map((i) => segs[i]);
}

/**
 * What was said that led to a decision, oldest first. A step opens the part of the conversation
 * it describes. A decision recorded without a chain has one step: its quote, with the lines just
 * before it.
 */
function DecisionChain({
  decision,
  segs,
  speakerName,
  onJump,
}: {
  decision: Decision;
  segs: TranscriptSegment[];
  speakerName: (id: string, saved: string) => string;
  onJump: (segIds: string[]) => void;
}) {
  const [open, setOpen] = useState<number | null>(null);
  const chain = decision.chain ?? [];
  let steps: { text: string; t: number; talk: TranscriptSegment[] }[];
  if (chain.length) {
    steps = chain.map((step) => ({ text: step.text, t: step.t, talk: talkOf(step, segs) }));
  } else if (decision.quote) {
    const at = segs.reduce((best, g, i) => (Math.abs(g.t_start - decision.t) < Math.abs(segs[best].t_start - decision.t) ? i : best), 0);
    steps = [{ text: `“${decision.quote}”`, t: decision.t, talk: segs.slice(Math.max(0, at - 2), at + 1) }];
  } else {
    return null;
  }
  return (
    <ol className="chain" aria-label="What led to this decision">
      {steps.map((step, i) => {
        const shown = open === i && step.talk.length > 0;
        return (
          <li key={i} className="chain-step" data-open={shown}>
            <button
              type="button"
              className="chain-btn"
              onClick={() => setOpen(shown ? null : i)}
              disabled={!step.talk.length}
              aria-expanded={shown}
              title={step.talk.length ? undefined : "This part of the transcript isn't available"}
            >
              <span className="chain-text">{step.text}</span>
              <span className="chain-t">{fmtT(step.t)}</span>
              {step.talk.length > 0 && <Icon name={shown ? "chevron-up" : "chevron-down"} />}
            </button>
            {shown && (
              <div className="chain-talk">
                <ol className="talk">
                  {step.talk.map((g) => (
                    <li key={g.seg_id} className="line">
                      <span className="tm">{fmtT(g.t_start)}</span>
                      <span className="who">{speakerName(g.speaker_id, g.speaker_name)}</span>
                      <span className="tx">{g.text}</span>
                    </li>
                  ))}
                </ol>
                <button type="button" className="jump" onClick={() => onJump(step.talk.map((g) => g.seg_id))}>
                  <Icon name="scroll-text" />
                  Show in the transcript
                </button>
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function ReportView({ meeting, onPushed }: { meeting: Meeting; onPushed: () => void }) {
  const agent = identity.agent_name;
  const { me, members, person } = useTeam();
  const report = useReport(meeting.id);
  const transcript = useSavedTranscript(meeting.id);
  const allDecisions = useDecisions();
  const meetings = useMeetings();
  const settings = useSettings();

  const [search, setSearch] = useState("");
  const [speaker, setSpeaker] = useState("all");
  const [highlighted, setHighlighted] = useState<string[]>([]);
  const highlightTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(highlightTimer.current), []);

  const segs = useMemo(() => transcript.data ?? [], [transcript.data]);
  const r = report.data;
  const start = meetingStart(meeting);
  const pushed = meeting.status === "pushed";
  // Review and pushing are about the tasks: a meeting that produced none has neither state.
  const reviewable = (r?.tasks.length ?? 0) > 0;

  // The saved name is the fallback for someone who has since left the team.
  const speakerName = (id: string, saved: string) =>
    id === AGENT_PARTICIPANT_ID ? agent : id === me.id ? me.short : (members.find((x) => x.id === id)?.short ?? shortOf(saved));

  const scrollTo = (domId: string, block: ScrollLogicalPosition = "start") =>
    document.getElementById(domId)?.scrollIntoView({ behavior: "smooth", block });

  /** Brings these transcript lines into view, in the transcript's own scroll, and marks them. */
  const jumpToLines = (segIds: string[]) => {
    if (!segIds.length) return;
    setSearch("");
    setSpeaker("all");
    setHighlighted(segIds);
    // Wait for the filters to clear so the lines are on the page.
    setTimeout(() => scrollTo(`seg-${segIds[0]}`, "center"), 0);
    clearTimeout(highlightTimer.current);
    highlightTimer.current = setTimeout(() => setHighlighted([]), 2600);
  };

  const jumpTo = (t: number) => {
    if (!segs.length) return;
    const nearest = segs.reduce((a, b) => (Math.abs(b.t_start - t) < Math.abs(a.t_start - t) ? b : a), segs[0]);
    jumpToLines([nearest.seg_id]);
  };

  const q = search.trim().toLowerCase();
  const filtered = segs.filter((g) => (speaker === "all" || g.speaker_id === speaker) && (!q || g.text.toLowerCase().includes(q)));
  const speakers = [...new Map(segs.map((g) => [g.speaker_id, speakerName(g.speaker_id, g.speaker_name)])).entries()];

  const mark = (text: string): ReactNode => {
    if (!q) return text;
    const out: ReactNode[] = [];
    const low = text.toLowerCase();
    let i = 0;
    while (i < text.length) {
      const j = low.indexOf(q, i);
      if (j < 0) {
        out.push(text.slice(i));
        break;
      }
      if (j > i) out.push(text.slice(i, j));
      out.push(<mark key={j}>{text.slice(j, j + q.length)}</mark>);
      i = j + q.length;
    }
    return out;
  };

  const toc: [string, string, number | ""][] = [
    ["r-summary", "Summary", ""],
    ["r-decisions", "Decisions", r?.decisions.length ?? ""],
    ["r-tasks", "Tasks", r?.tasks.length ?? ""],
    ["r-transcript", "Transcript", segs.length || ""],
  ];

  const people = joinNames(attendeeNames(meeting, (id) => person(id).short, agent));
  const timeText = start
    ? meeting.duration_min
      ? `${fmtClock(start)}–${fmtClock(new Date(start.getTime() + meeting.duration_min * 60_000))} · ${meeting.duration_min} min`
      : fmtClock(start)
    : null;

  return (
    <>
      <BackLink />
      <div className="rep-grid">
        <nav className="toc" aria-label="Report sections">
          {toc.map(([id, label, count]) => (
            <button key={id} type="button" onClick={() => scrollTo(id)}>
              {label}
              <span className="n">{count}</span>
            </button>
          ))}
        </nav>
        <article className="rep">
          <header className="rcard">
            {reviewable && (
              <span className="rep-status" data-pushed={pushed}>
                <Icon name={pushed ? "circle-check" : "file-pen-line"} />
                {pushed ? "Pushed" : "Needs review"}
              </span>
            )}
            <h1 className="rep-title">{meeting.title}</h1>
            <div className="rep-meta">
              {start && (
                <span>
                  <Icon name="calendar" />
                  {fmtLongDate(start)}
                </span>
              )}
              {timeText && (
                <span>
                  <Icon name="clock" />
                  {timeText}
                </span>
              )}
              <span>
                <Icon name="users" />
                {people}
              </span>
            </div>
          </header>

          {report.error && !r && (
            <section className="rcard">
              <Notice error={report.error} onRetry={report.reload}>
                The report can&apos;t be loaded.
              </Notice>
            </section>
          )}
          {report.loading && !r && (
            <section className="rcard">
              <p className="muted-p">Loading the report…</p>
            </section>
          )}

          {r && (
            <>
              <section className="rcard" aria-labelledby="r-summary">
                <div className="sec-row">
                  <h2 id="r-summary" className="sec-h">
                    Summary
                  </h2>
                  {r.summary && <ListenButton meetingId={meeting.id} />}
                </div>
                {r.summary ? <p className="summary">{r.summary}</p> : <p className="muted-p">No summary was written.</p>}
              </section>

              <section className="rcard" aria-labelledby="r-decisions">
                <h2 id="r-decisions" className="sec-h">
                  Decisions
                </h2>
                {r.decisions.length === 0 && <p className="muted-p">No decisions were recorded.</p>}
                {r.decisions.map((d, i) => {
                  const related = d.relation ? allDecisions.data?.find((x) => x.id === d.relation?.decision_id) : undefined;
                  const relatedMeeting = related ? meetings.data?.find((x) => x.id === related.meeting_id) : undefined;
                  const relatedStart = relatedMeeting ? meetingStart(relatedMeeting) : null;
                  return (
                    <div key={d.id} className="decision">
                      <span className="dnum">{String(i + 1).padStart(2, "0")}</span>
                      <div>
                        <p className="dtext">{d.text}</p>
                        <DecisionChain decision={d} segs={segs} speakerName={speakerName} onJump={jumpToLines} />
                        {d.relation && (
                          <div className="contra" role="note">
                            <Icon name="git-compare" />
                            <div style={{ minWidth: 0 }}>
                              <div className="contra-h">
                                {d.relation.type === "contradicts" ? "Contradicts a past decision" : "Superseded by a later decision"}
                              </div>
                              {related && <p>“{related.text}”</p>}
                              <Link className="link" href="/memory">
                                {[relatedMeeting?.title, relatedStart && fmtDate(relatedStart, true), "see in Decisions"]
                                  .filter(Boolean)
                                  .join(" · ")}
                              </Link>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </section>

              <section className="rcard" aria-labelledby="r-tasks">
                <h2 id="r-tasks" className="sec-h">
                  Tasks
                </h2>
                <Tasks
                  key={meeting.id}
                  meeting={meeting}
                  report={r}
                  settings={settings.data}
                  onPushed={onPushed}
                  onJump={segs.length ? jumpTo : null}
                />
              </section>

              <Bullets id="r-blockers" title="Blockers" icon="triangle-alert" items={r.blockers} />
              {r.links.length > 0 && (
                <section className="rcard" aria-labelledby="r-links">
                  <h2 id="r-links" className="sec-h">
                    Links
                  </h2>
                  <Sources sources={r.links} />
                </section>
              )}
            </>
          )}

          <section className="rcard" aria-labelledby="r-transcript">
            <h2 id="r-transcript" className="sec-h">
              Transcript
            </h2>
            {transcript.error && !transcript.data ? (
              <Notice error={transcript.error} onRetry={transcript.reload}>
                The transcript can&apos;t be loaded.
              </Notice>
            ) : transcript.loading && !transcript.data ? (
              <p className="muted-p">Loading the transcript…</p>
            ) : !segs.length ? (
              <p className="muted-p">No transcript was saved for this meeting.</p>
            ) : (
              <>
                <div className="t-tools">
                  <label className="t-search">
                    <Icon name="search" />
                    <input
                      className="field"
                      type="search"
                      aria-label="Search transcript"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search the transcript"
                    />
                  </label>
                  <div className="chips" role="group" aria-label="Filter by speaker">
                    {[["all", "Everyone"], ...speakers].map(([id, label]) => (
                      <button key={id} type="button" className="chip" onClick={() => setSpeaker(id)} aria-pressed={speaker === id}>
                        {label}
                      </button>
                    ))}
                  </div>
                  <span className="t-count">
                    {filtered.length} of {segs.length} lines
                  </span>
                </div>
                {/* Its own scroll: a long meeting does not stretch the page. */}
                <ol className="lines" tabIndex={0} aria-label="Transcript lines">
                  {filtered.map((g) => (
                    <li key={g.seg_id} id={`seg-${g.seg_id}`} className="line" data-hl={highlighted.includes(g.seg_id)}>
                      <span className="tm">{fmtT(g.t_start)}</span>
                      <span className="who">{speakerName(g.speaker_id, g.speaker_name)}</span>
                      <span className="tx">{mark(g.text)}</span>
                    </li>
                  ))}
                </ol>
                {!filtered.length && <p className="muted-p">No lines match.</p>}
              </>
            )}
          </section>
        </article>
      </div>
    </>
  );
}

/** The drafts as the reviewer is editing them, and the one approved push to Jira or GitHub. */
function Tasks({
  meeting,
  report,
  settings,
  onPushed,
  onJump,
}: {
  meeting: Meeting;
  report: Report;
  settings: TeamSettings | undefined;
  onPushed: () => void;
  onJump: ((t: number) => void) | null;
}) {
  const { me, members } = useTeam();
  const [tasks, setTasks] = useState<TaskDraft[]>(report.tasks);
  const [destination, setDestination] = useState<TaskDestination>("jira");
  const [pushing, setPushing] = useState(false);
  const [justPushed, setJustPushed] = useState(false);
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [problem, setProblem] = useState("");

  const pushed = meeting.status === "pushed" || justPushed;
  const included = tasks.filter((t) => t.include);
  const jira = settings?.jira;
  const github = settings?.github;
  const destinationLabel = destination === "jira" ? (jira?.project ?? "Jira") : (github?.repos[0]?.path ?? "GitHub");
  const keys = tasks.flatMap((t) => (t.key ? [t.key] : []));

  const commit = (task: TaskDraft) => {
    updateTask(meeting.id, task).then(
      () => setProblem(""),
      (err: unknown) => setProblem(`Your edit isn't saved. ${describeError(err)}`),
    );
  };

  const push = async () => {
    setPushing(true);
    setProblem("");
    try {
      // Save the drafts as they stand, then send exactly the ones that were ticked.
      await Promise.all(tasks.map((t) => updateTask(meeting.id, t)));
      const results = await pushTasks(meeting.id, { task_ids: included.map((t) => t.id), destination, approved_by: me.id });
      const failed = results.filter((x) => x.error);
      setTasks((list) => list.map((t) => ({ ...t, key: results.find((x) => x.task_id === t.id)?.key ?? t.key })));
      setUrls((prev) => ({ ...prev, ...Object.fromEntries(results.flatMap((x) => (x.url ? [[x.task_id, x.url]] : []))) }));
      if (failed.length) {
        setProblem(`${failed.length} of ${results.length} weren't pushed: ${failed.map((x) => x.error).join("; ")}`);
      } else {
        setJustPushed(true);
      }
      onPushed();
    } catch (err) {
      setProblem(`Nothing was pushed. ${describeError(err)}`);
    } finally {
      setPushing(false);
    }
  };

  const status = problem
    ? problem
    : pushed
      ? keys.length
        ? `${keys.join(", ")} in ${destinationLabel}.`
        : ""
      : included.length === 0
        ? "Select at least one task."
        : `${included.length} of ${tasks.length} selected. Nothing is sent until you press Push.`;

  return (
    <TaskReview
      tasks={tasks}
      members={members}
      locked={pushed || pushing}
      pushing={pushing}
      pushed={pushed}
      destination={destination}
      destinationLabel={destinationLabel}
      keyUrl={(task) => urls[task.id] ?? (destination === "jira" ? jiraIssueUrl(jira?.site, task.key) : null)}
      status={status}
      onChange={(task) => setTasks((list) => list.map((t) => (t.id === task.id ? task : t)))}
      onCommit={commit}
      onDestination={setDestination}
      onPush={() => void push()}
      onJump={onJump}
      canPush={hostOrAdmin(meeting, me)}
    />
  );
}
