"use client";

import { type Meeting, type Source, identity } from "@moe/contracts";
import Link from "next/link";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Avatar } from "@/components/ui/Avatar";
import { Icon, type IconName } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { Notice } from "@/components/ui/Notice";
import { Sources, Unavailable } from "@/components/ui/Sources";
import type { Query } from "@/hooks/useApi";
import { askHistory, describeError } from "@/lib/api";
import { STATUS_LABEL, isSameDay, meetingStart, meetingTimeText } from "@/lib/format";

const STATUS_ICON: Record<Meeting["status"], IconName> = {
  scheduled: "calendar-clock",
  live: "radio",
  processing: "loader-circle",
  needs_review: "file-pen-line",
  pushed: "circle-check",
};

const PROMPTS: [string, IconName][] = [
  ["What did we decide in the last meeting?", "git-compare"],
  ["What's on my plate this week?", "list-checks"],
  ["Catch me up on the most recent meeting", "scroll-text"],
];

interface ChatEntry {
  id: string;
  from: "user" | "agent";
  text: string;
  sources?: Source[];
  unavailable?: string[];
  /** A reply saying the question failed; not sent back as conversation. */
  failed?: boolean;
}

const startMs = (m: Meeting) => meetingStart(m)?.getTime() ?? 0;

function MeetingCard({ meeting, now }: { meeting: Meeting; now: Date }) {
  const { person } = useTeam();
  const host = person(meeting.host_id);
  const others = meeting.participant_ids.filter((id) => id !== meeting.host_id).length;
  const live = meeting.status === "live";
  const scheduled = meeting.status === "scheduled";
  const timeText = meetingTimeText(meeting, now);
  const label = STATUS_LABEL[meeting.status];
  const opens = live ? "Join" : scheduled ? "Open lobby" : "Open report";
  return (
    <Link
      className={live ? "mcard live" : "mcard"}
      href={live || scheduled ? `/m/${meeting.code}` : `/meetings/${meeting.id}`}
      aria-label={`${meeting.title}, ${timeText}, ${label}. ${opens}`}
    >
      <Avatar person={host} size="lg" />
      <span className="mcard-main">
        {live && <span className="live-dot">Live now</span>}
        <span className="mcard-title">{meeting.title}</span>
        <span className="mcard-meta">
          {timeText && <span className="mcard-time">{timeText}</span>}
          <span>
            {host.short}
            {scheduled ? " (organiser)" : ""}
          </span>
          {others > 0 && (
            <span>
              +{others} other{others === 1 ? "" : "s"}
            </span>
          )}
        </span>
      </span>
      {live ? (
        <span className="join-pill">
          Join
          <Icon name="arrow-right" />
        </span>
      ) : (
        <span className="status-ic has-tip">
          <Icon name={STATUS_ICON[meeting.status]} />
          <span role="tooltip" className="tip">
            {label}
          </span>
        </span>
      )}
    </Link>
  );
}

/** Today's and earlier meetings, with a box to ask the agent about any of them. */
export function Rail({ meetings, now }: { meetings: Query<Meeting[]>; now: Date }) {
  const agent = identity.agent_name;
  const [panel, setPanel] = useState<"meetings" | "earlier" | "chat">("meetings");
  const [suggest, setSuggest] = useState(false);
  const [input, setInput] = useState("");
  const [chat, setChat] = useState<ChatEntry[]>([]);
  const [typing, setTyping] = useState(false);
  const log = useRef<HTMLDivElement>(null);
  const turn = useRef(0);
  // Bumped when the chat is cleared, so answers still on their way are dropped.
  const epoch = useRef(0);
  const pending = useRef(0);

  useEffect(() => {
    if (log.current) log.current.scrollTop = log.current.scrollHeight;
  }, [chat, typing, panel]);

  const all = meetings.data ?? [];
  const today = all
    .filter((m) => m.status === "live" || (m.status === "scheduled" && isSameDay(meetingStart(m) ?? now, now)))
    .sort((a, b) => startMs(a) - startMs(b));
  const upcoming = all
    .filter((m) => m.status === "scheduled" && !today.includes(m))
    .sort((a, b) => startMs(a) - startMs(b));
  const earlier = all
    .filter((m) => m.status !== "live" && m.status !== "scheduled")
    .sort((a, b) => startMs(b) - startMs(a));

  const ask = async (text: string) => {
    const question = text.trim();
    if (!question) return;
    const id = ++turn.current;
    const asked = epoch.current;
    pending.current += 1;
    setPanel("chat");
    setSuggest(false);
    setInput("");
    setTyping(true);
    // Earlier turns go with the question, so a follow-up has its context.
    const history = chat.filter((c) => !c.failed).map((c) => ({ role: c.from, text: c.text }));
    setChat((c) => [...c, { id: `q${id}`, from: "user", text: question }]);
    let reply: ChatEntry;
    try {
      const answer = await askHistory({ question, visibility: "private", history });
      reply = { id: `a${id}`, from: "agent", text: answer.text, sources: answer.sources, unavailable: answer.unavailable };
    } catch (err) {
      reply = { id: `a${id}`, from: "agent", text: `I couldn't answer that. ${describeError(err)}`, failed: true };
    }
    if (epoch.current !== asked) return;
    pending.current -= 1;
    setChat((c) => [...c, reply]);
    setTyping(pending.current > 0);
  };

  const clearChat = () => {
    epoch.current += 1;
    pending.current = 0;
    setChat([]);
    setTyping(false);
    setPanel("meetings");
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void ask(input);
  };

  const label = panel === "chat" ? `Ask ${agent}` : panel === "earlier" ? "Earlier meetings" : "Today";

  return (
    <aside className="rail" aria-label={label}>
      {panel === "chat" ? (
        <>
          <div className="chat-head">
            <button type="button" className="icon-btn sm" onClick={() => setPanel("meetings")} aria-label="Back to meetings">
              <Icon name="arrow-left" />
            </button>
            <span className="mark-slot">
              <Mark size={26} tile />
            </span>
            <h2 className="rail-h">Ask {agent}</h2>
            <button type="button" className="btn btn-quiet btn-sm" style={{ marginLeft: "auto" }} onClick={clearChat}>
              <Icon name="rotate-ccw" />
              New chat
            </button>
          </div>
          <div ref={log} className="chat-log" role="log" aria-live="polite" aria-label={`Conversation with ${agent}`}>
            {chat.map((c) =>
              c.from === "user" ? (
                <p key={c.id} className="bubble-user">
                  {c.text}
                </p>
              ) : (
                <div key={c.id} className="bubble-agent">
                  <span className="mark-slot">
                    <Mark size={26} tile />
                  </span>
                  <div className="bubble-agent-body">
                    <p>{c.text}</p>
                    <Sources sources={c.sources ?? []} />
                    <Unavailable items={c.unavailable ?? []} />
                  </div>
                </div>
              ),
            )}
            {typing && (
              <div className="typing">
                <span className="mark-slot">
                  <Mark size={26} tile state="working" />
                </span>
                Looking through your meetings…
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="rail-scroll">
          {panel === "earlier" ? (
            <>
              <div className="rail-head">
                <div className="rail-back">
                  <button type="button" className="icon-btn sm" onClick={() => setPanel("meetings")} aria-label="Back to today">
                    <Icon name="arrow-left" />
                  </button>
                  <h2 className="rail-h">Earlier</h2>
                </div>
              </div>
              <section className="rail-group" aria-label="Earlier meetings">
                {earlier.map((m) => (
                  <MeetingCard key={m.id} meeting={m} now={now} />
                ))}
                {meetings.data && !earlier.length && <p className="rail-empty">No earlier meetings yet.</p>}
              </section>
            </>
          ) : (
            <>
              <section className="rail-group" aria-label="Today">
                <div className="rail-head">
                  <h2 className="rail-h">Today</h2>
                  <button type="button" className="btn btn-quiet btn-sm" onClick={() => setPanel("earlier")} aria-label="Earlier meetings">
                    <Icon name="history" />
                    Earlier
                  </button>
                </div>
                {today.map((m) => (
                  <MeetingCard key={m.id} meeting={m} now={now} />
                ))}
                {meetings.data && !today.length && <p className="rail-empty">No live or scheduled meetings today.</p>}
              </section>
              {upcoming.length > 0 && (
                <section className="rail-group" aria-label="Upcoming">
                  <div className="rail-head">
                    <h2 className="rail-h">Upcoming</h2>
                  </div>
                  {upcoming.map((m) => (
                    <MeetingCard key={m.id} meeting={m} now={now} />
                  ))}
                </section>
              )}
            </>
          )}
          {meetings.loading && !meetings.data && <p className="rail-empty">Loading meetings…</p>}
          {meetings.error && (
            <Notice error={meetings.error} onRetry={meetings.reload}>
              Meetings can&apos;t be loaded.
            </Notice>
          )}
        </div>
      )}

      <form className="ask" onSubmit={onSubmit}>
        {suggest && (
          <div className="suggest" role="menu" aria-label="Suggested questions">
            <span className="suggest-h">Try asking</span>
            {PROMPTS.map(([text, icon]) => (
              <button key={text} type="button" role="menuitem" className="suggest-item" onClick={() => void ask(text)}>
                <Icon name={icon} />
                {text}
              </button>
            ))}
          </div>
        )}
        <div className="ask-bar">
          <button
            type="button"
            className="icon-btn ask-spark"
            onClick={() => setSuggest((open) => !open)}
            aria-expanded={suggest}
            aria-label="Suggested questions"
            title="Suggested questions"
          >
            <Icon name="sparkles" />
          </button>
          <input
            aria-label={`Ask ${agent} about your meetings`}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={panel === "chat" ? "Ask a follow-up" : "Ask about past meetings, tasks, decisions…"}
          />
          <button type="submit" className="send-btn" aria-label="Send">
            <Icon name="arrow-up" />
          </button>
        </div>
      </form>
    </aside>
  );
}
