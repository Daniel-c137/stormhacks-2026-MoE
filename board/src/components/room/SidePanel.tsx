"use client";

import {
  AGENT_PARTICIPANT_ID,
  type Agenda,
  type ChatMessage,
  type CodeSnippet,
  type FactCheck,
  type Participant,
  type Person,
  type Source,
  identity,
  mention,
} from "@moe/contracts";
import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Avatar } from "@/components/ui/Avatar";
import { Icon } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { Sources, Unavailable } from "@/components/ui/Sources";
import { useDismiss } from "@/hooks/useDismiss";
import { fmtClock, initialsOf } from "@/lib/format";
import { FactCheckBody } from "./FactCheckView";
import { LiveAgenda } from "./LiveAgenda";

/** A chat message plus what an agent answer carries with it. */
export interface PanelMessage extends ChatMessage {
  sources?: Source[];
  unavailable?: string[];
  snippet?: CodeSnippet | null;
}

export interface SidePanelProps {
  messages: PanelMessage[];
  me: Person;
  /** The other people in the room, for private messages. */
  people: Participant[];
  personOf: (participant: Participant) => Person;
  agentTyping: boolean;
  error: string;
  agenda: Agenda | null;
  agendaError: Error | null;
  /** This meeting's fact-checks this participant may see, oldest first. */
  checks: FactCheck[];
  /** to=null sends to everyone; otherwise a private message to that participant (or the agent). */
  onSend: (text: string, to: string | null) => void;
  onClose: () => void;
}

const PREVIEW_LINES = 4;

type Tab = "chat" | "agenda" | "checks";
const TABS: [Tab, string][] = [
  ["chat", "Chat"],
  ["agenda", "Agenda"],
  ["checks", "Fact-checks"],
];

function ChatSnippet({ snippet }: { snippet: CodeSnippet }) {
  const [expanded, setExpanded] = useState(false);
  const lines = snippet.code.split("\n");
  const from = snippet.highlight ? Math.max(snippet.highlight[0] - snippet.start_line, 0) : 0;
  const shown = expanded ? snippet.code : lines.slice(from, from + PREVIEW_LINES).join("\n");
  return (
    <div className="code">
      <div className="code-head">
        <Icon name="file-code" />
        <span className="path">{snippet.path}</span>
        <span className="rng">
          L{snippet.start_line}–{snippet.end_line}
        </span>
      </div>
      <pre>{shown}</pre>
      <div className="code-foot">
        <button type="button" onClick={() => setExpanded((x) => !x)} aria-expanded={expanded}>
          <Icon name={expanded ? "chevron-up" : "chevron-down"} />
          {expanded ? "Show less" : `Show all ${lines.length} lines`}
        </button>
        <a href={snippet.github_url} target="_blank" rel="noopener noreferrer">
          <Icon name="github" />
          GitHub
        </a>
      </div>
    </div>
  );
}

/** The meeting's fact-checks, newest first. */
function CheckList({ checks, me }: { checks: FactCheck[]; me: Person }) {
  const agent = identity.agent_name;
  if (!checks.length) return <p className="msgs-empty">No fact-checks yet. {agent} checks technical claims as the meeting goes.</p>;
  return (
    <ul className="fc-list">
      {[...checks].reverse().map((c) => (
        <li key={c.id} className="fc-item" data-hand={c.raised_hand}>
          <FactCheckBody check={c} meId={me.id} showTime />
        </li>
      ))}
    </ul>
  );
}

/** Meeting chat (public messages, private messages, private questions to the agent), the live
 * agenda, and the meeting's fact-checks. */
export function SidePanel({
  messages,
  me,
  people,
  personOf,
  agentTyping,
  error,
  agenda,
  agendaError,
  checks,
  onSend,
  onClose,
}: SidePanelProps) {
  const agent = identity.agent_name;
  const [tab, setTab] = useState<Tab>("chat");
  const [text, setText] = useState("");
  const [to, setTo] = useState<string | null>(null);
  const [toOpen, setToOpen] = useState(false);
  const log = useRef<HTMLDivElement>(null);
  const toWrap = useRef<HTMLDivElement>(null);
  useDismiss(
    toWrap,
    toOpen,
    useCallback(() => setToOpen(false), []),
  );

  useEffect(() => {
    if (log.current) log.current.scrollTop = log.current.scrollHeight;
  }, [messages.length, agentTyping, tab]);

  const byId = new Map(people.map((p) => [p.id, p]));
  const nameOf = (id: string) => {
    if (id === AGENT_PARTICIPANT_ID) return agent;
    const p = byId.get(id);
    return p ? personOf(p).short : "someone who left";
  };
  // A private recipient who has left the room can no longer be messaged.
  const target = to && (to === AGENT_PARTICIPANT_ID || byId.has(to)) ? to : null;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const body = text.trim();
    if (!body) return;
    onSend(body, target);
    setText("");
  };

  const pick = (id: string | null) => () => {
    setTo(id);
    setToOpen(false);
  };

  return (
    <aside className="chatcard" aria-label="Meeting panel">
      <div className="chat-tabs" role="tablist" aria-label="Panel">
        {TABS.map(([id, label]) => (
          <button
            key={id}
            type="button"
            className="chat-tab"
            role="tab"
            id={`tab-${id}`}
            aria-controls={`panel-${id}`}
            aria-selected={tab === id}
            onClick={() => setTab(id)}
          >
            {label}
            {id === "checks" && checks.length > 0 && <span className="tab-n">{checks.length}</span>}
          </button>
        ))}
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Hide panel">
          <Icon name="panel-right-close" />
        </button>
      </div>
      {tab === "agenda" && (
        <div className="msgs" role="tabpanel" id="panel-agenda" aria-labelledby="tab-agenda">
          <LiveAgenda agenda={agenda} error={agendaError} />
        </div>
      )}
      {tab === "checks" && (
        <div className="msgs" role="tabpanel" id="panel-checks" aria-labelledby="tab-checks">
          <CheckList checks={checks} me={me} />
        </div>
      )}
      <div className="chat-pane" role="tabpanel" id="panel-chat" aria-labelledby="tab-chat" hidden={tab !== "chat"}>
        <div ref={log} className="msgs" role="log" aria-live="polite" aria-label="Chat messages">
          {messages.length === 0 && !agentTyping && (
            <p className="msgs-empty">
              No messages yet. Write to everyone, or mention {mention()} to ask a question the whole room can see.
            </p>
          )}
          {messages.map((m) => {
            const mine = m.sender_id === me.id;
            const sender = byId.get(m.sender_id);
            const who = mine ? "You" : m.is_agent ? agent : sender ? personOf(sender).short : m.sender_name;
            return (
              <div key={m.id} className="msg" data-from={mine ? "me" : m.is_agent ? "agent" : "other"}>
                {m.is_agent ? (
                  <span className="mark-slot">
                    <Mark size={28} tile />
                  </span>
                ) : (
                  <Avatar
                    person={mine ? me : sender ? personOf(sender) : { name: m.sender_name, initials: initialsOf(m.sender_name) }}
                    me={mine}
                  />
                )}
                <div className="msg-body">
                  <div className="msg-head">
                    <b>{who}</b>
                    {m.recipient_id && (
                      <span className="priv">
                        <Icon name="lock" />
                        {m.recipient_id === me.id ? "Private · to you" : `Private · to ${nameOf(m.recipient_id)}`}
                      </span>
                    )}
                    <span className="msg-ts">{fmtClock(new Date(m.ts))}</span>
                  </div>
                  <p className="msg-text">{m.text}</p>
                  {m.snippet && <ChatSnippet snippet={m.snippet} />}
                  <Sources sources={m.sources ?? []} newTab />
                  <Unavailable items={m.unavailable ?? []} />
                </div>
              </div>
            );
          })}
          {agentTyping && (
            <div className="typing">
              <span className="mark-slot">
                <Mark size={28} tile state="working" />
              </span>
              {agent} is looking into it…
            </div>
          )}
        </div>
        <form className="compose" onSubmit={submit}>
          <div className="to-wrap" ref={toWrap}>
            <button
              type="button"
              className="to-btn"
              onClick={() => setToOpen((open) => !open)}
              aria-haspopup="listbox"
              aria-expanded={toOpen}
              aria-label={`Send to: ${target ? `${nameOf(target)}, private` : "Everyone"}`}
              data-private={Boolean(target)}
            >
              {target && <Icon name="lock" />}
              <span className="k">To</span>
              <b>{target ? nameOf(target) : "Everyone"}</b>
              <Icon name={toOpen ? "chevron-up" : "chevron-down"} />
            </button>
            {toOpen && (
              <div className="menu" role="listbox" aria-label="Send to">
                <button type="button" className="menu-item" role="option" aria-selected={!target} onClick={pick(null)}>
                  <span className="av sm av-me">
                    <Icon name="users" />
                  </span>
                  <span className="grow-name">Everyone</span>
                  <Icon name="check" />
                </button>
                <div className="menu-label">Private message</div>
                <button
                  type="button"
                  className="menu-item"
                  role="option"
                  aria-selected={target === AGENT_PARTICIPANT_ID}
                  onClick={pick(AGENT_PARTICIPANT_ID)}
                >
                  <span className="mark-slot">
                    <Mark size={26} tile />
                  </span>
                  <span className="grow-name">{agent}</span>
                  <Icon name="check" />
                </button>
                {people.map((p) => (
                  <button key={p.id} type="button" className="menu-item" role="option" aria-selected={target === p.id} onClick={pick(p.id)}>
                    <Avatar person={personOf(p)} size="sm" />
                    <span className="grow-name">{personOf(p).name}</span>
                    <Icon name="check" />
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="compose-row">
            <input
              className="field"
              aria-label="Message"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={
                !target
                  ? `Message everyone, or ${mention()} to ask`
                  : target === AGENT_PARTICIPANT_ID
                    ? `Ask ${agent} privately`
                    : `Private message to ${nameOf(target)}`
              }
            />
            <button type="submit" className="send-btn" aria-label="Send">
              <Icon name="send" />
            </button>
          </div>
          <span role="alert" className="err">
            {error}
          </span>
        </form>
      </div>
    </aside>
  );
}
