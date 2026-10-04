"use client";

import { type Agenda, type AgendaItemInput, type AgendaSuggestions, identity } from "@moe/contracts";
import { type FormEvent, type KeyboardEvent, useCallback, useRef, useState } from "react";
import { Icon, Spinner } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { useAgenda } from "@/hooks/useApi";
import { useDismiss } from "@/hooks/useDismiss";
import { describeError, rewriteAgendaItem, suggestAgenda, updateAgenda } from "@/lib/api";
import { agendaTitleKey, draftedTopics } from "@/lib/agenda";

/** A topic as edited here. `key` is stable for React; `id` is the brain's, once it has saved it. */
interface Topic extends AgendaItemInput {
  key: string;
}

/** What the last draft added, for the confirmation and its Undo. */
interface Drafted {
  keys: string[];
  sources: string[];
  unavailable: string[];
}

const MINUTES_MIN = 1;
const MINUTES_MAX = 240;

const fromAgenda = (agenda: Agenda): Topic[] =>
  agenda.items.map((item) => ({ key: item.id, id: item.id, title: item.title, minutes: item.minutes ?? null }));

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** Topics for the meeting. People add their own, or ask the agent to draft some from the team's open
 * work; either way they become the same ordinary items: rename, timebox, reorder, reword, delete.
 * Saved with the meeting (PUT /meetings/{id}/agenda takes the whole list; items without an id are new). */
export function LobbyAgenda({ meetingId }: { meetingId: string }) {
  const agent = identity.agent_name;
  const agenda = useAgenda(meetingId);
  // Edits made here; until the first one, the saved agenda is shown as it is.
  const [edited, setEdited] = useState<Topic[] | null>(null);
  const items = edited ?? (agenda.data ? fromAgenda(agenda.data) : []);
  const latest = useRef(items);
  latest.current = items;
  const [input, setInput] = useState("");
  const [rewriting, setRewriting] = useState(false);
  const [rewrote, setRewrote] = useState(false);
  const [rewording, setRewording] = useState<string | null>(null);
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [timing, setTiming] = useState<string | null>(null);
  const [timeDraft, setTimeDraft] = useState("");
  // The person chose "Add your own items" on an empty agenda.
  const [manual, setManual] = useState(false);
  const [drafting, setDrafting] = useState(false);
  const [drafted, setDrafted] = useState<Drafted | null>(null);
  const [error, setError] = useState("");
  const list = useRef<HTMLUListElement>(null);
  // The brain's id for each topic it has saved, by key. Saves run one at a time, so a topic added
  // while an earlier save is in flight is sent with its id once that save has returned.
  const ids = useRef(new Map<string, string>());
  const queue = useRef<Promise<void>>(Promise.resolve());
  useDismiss(
    list,
    menuFor !== null,
    useCallback(() => setMenuFor(null), []),
  );

  const commit = (next: Topic[]) => {
    setEdited(next);
    latest.current = next;
    setError("");
    queue.current = queue.current.then(async () => {
      const body = {
        items: next.map((t) => ({ id: t.id ?? ids.current.get(t.key) ?? null, title: t.title, minutes: t.minutes ?? null })),
      };
      try {
        const saved = await updateAgenda(meetingId, body);
        // The brain keeps the order it was sent, so the nth saved item is the nth topic.
        next.forEach((t, i) => {
          const savedId = saved.items[i]?.id;
          if (savedId) ids.current.set(t.key, savedId);
        });
      } catch (err) {
        setError(`These topics aren't saved yet. ${describeError(err)}`);
      }
    });
  };

  /** A person's edit. Drafted items are theirs to change now, so the draft's Undo goes away. */
  const save = (next: Topic[]) => {
    setDrafted(null);
    commit(next);
  };

  const hasText = Boolean(input.trim());
  const loaded = Boolean(agenda.data || agenda.error);
  const choosing = loaded && items.length === 0 && !manual;

  const add = (e: FormEvent) => {
    e.preventDefault();
    const title = input.trim();
    if (!title || rewriting) return;
    save([...items, { key: crypto.randomUUID(), title }]);
    setInput("");
    setRewrote(false);
  };

  const rewrite = async () => {
    const text = input.trim();
    if (!text) return;
    setRewriting(true);
    setError("");
    try {
      const result = await rewriteAgendaItem({ text });
      setInput(result.text);
      setRewrote(true);
    } catch (err) {
      setError(`${agent} couldn't rewrite that. ${describeError(err)}`);
    } finally {
      setRewriting(false);
    }
  };

  /** Rewords a topic on the list into the edit field, for the person to keep (Enter) or drop (Esc). */
  const reword = async (topic: Topic) => {
    setRewording(topic.key);
    setError("");
    try {
      const result = await rewriteAgendaItem({ text: topic.title });
      setEditing(topic.key);
      setDraft(result.text);
    } catch (err) {
      setError(`${agent} couldn't reword that. ${describeError(err)}`);
    } finally {
      setRewording(null);
    }
  };

  /** Only on click: the agent proposes items from the team's open work and recent meetings, and
   * they are added after what is there. Nothing people wrote is replaced or rewritten. */
  const draftWithAgent = async () => {
    if (drafting) return;
    setDrafting(true);
    setDrafted(null);
    setError("");
    setMenuFor(null);
    let result: AgendaSuggestions;
    try {
      // The brain leaves out what is already saved, so let pending edits land first.
      await queue.current;
      result = await suggestAgenda(meetingId);
    } catch (err) {
      setError(`${agent} couldn't draft the agenda. ${describeError(err)}`);
      setDrafting(false);
      return;
    }
    const current = latest.current;
    const added: Topic[] = draftedTopics(current, result.items).map((item) => ({
      key: crypto.randomUUID(),
      title: item.title,
      minutes: item.minutes ?? null,
    }));
    if (added.length) commit([...current, ...added]);
    const kept = new Set(added.map((t) => agendaTitleKey(t.title)));
    const sources = result.items
      .filter((i) => kept.has(agendaTitleKey(i.title)))
      .flatMap((i) => i.sources.map((s) => s.label));
    setDrafted({ keys: added.map((t) => t.key), sources: [...new Set(sources)], unavailable: result.unavailable });
    setDrafting(false);
  };

  const undoDraft = () => {
    if (!drafted) return;
    const keys = new Set(drafted.keys);
    setDrafted(null);
    commit(latest.current.filter((t) => !keys.has(t.key)));
  };

  const saveEdit = (key: string) => {
    if (editing !== key) return;
    const title = draft.trim();
    const topic = items.find((x) => x.key === key);
    if (title && topic && title !== topic.title) save(items.map((x) => (x.key === key ? { ...x, title } : x)));
    setEditing(null);
  };

  const saveTime = (key: string) => {
    if (timing !== key) return;
    setTiming(null);
    const text = timeDraft.trim();
    const minutes = text ? Number(text) : null;
    if (minutes !== null && !(Number.isInteger(minutes) && minutes >= MINUTES_MIN && minutes <= MINUTES_MAX)) {
      setError(`A timebox is ${MINUTES_MIN} to ${MINUTES_MAX} minutes, or none.`);
      return;
    }
    const topic = items.find((x) => x.key === key);
    if (topic && (topic.minutes ?? null) !== minutes) save(items.map((x) => (x.key === key ? { ...x, minutes } : x)));
  };

  const move = (key: string, by: -1 | 1) => {
    const from = items.findIndex((x) => x.key === key);
    const to = from + by;
    if (from < 0 || to < 0 || to >= items.length) return;
    const next = [...items];
    [next[from], next[to]] = [next[to], next[from]];
    save(next);
  };

  const onFieldKey = (e: KeyboardEvent<HTMLInputElement>, cancel: () => void) => {
    if (e.key === "Enter") {
      e.preventDefault();
      e.currentTarget.blur();
    }
    if (e.key === "Escape") {
      e.preventDefault();
      cancel();
    }
  };

  const draftButton = (
    <button type="button" className="btn btn-outline btn-sm agenda-draft-btn" onClick={() => void draftWithAgent()} disabled={drafting}>
      {drafting ? <Spinner size={15} /> : <Mark size={15} />}
      {drafting ? `${agent} is drafting…` : `Draft with ${agent}`}
    </button>
  );

  return (
    <section className="agenda" aria-labelledby="h-agenda" aria-busy={drafting}>
      <h2 id="h-agenda">What do you want to talk about?</h2>
      <div className="agenda-scroll">
        {choosing ? (
          <div className="agenda-start" role="group" aria-label="Start the agenda">
            <button type="button" className="agenda-choice" onClick={() => setManual(true)} disabled={drafting}>
              <span className="agenda-choice-ic">
                <Icon name="plus" />
              </span>
              <span className="agenda-choice-title">Add your own items</span>
              <span className="agenda-choice-text">Type the topics you want to cover.</span>
            </button>
            <button type="button" className="agenda-choice" onClick={() => void draftWithAgent()} disabled={drafting}>
              <span className="agenda-choice-ic">{drafting ? <Spinner size={18} /> : <Mark size={18} />}</span>
              <span className="agenda-choice-title">{drafting ? `${agent} is drafting…` : `Let ${agent} draft it`}</span>
              <span className="agenda-choice-text">
                {drafting
                  ? "Reading open work and recent meetings."
                  : "From open Jira work and recent meetings. You can change everything."}
              </span>
            </button>
          </div>
        ) : (
          <>
            <ul className="agenda-list" ref={list}>
              {items.map((item, index) => {
                const menuLabel = `Options for “${item.title}”`;
                return (
                  <li key={item.key} className="agenda-item">
                    {editing === item.key ? (
                      <input
                        className="field"
                        aria-label="Edit topic"
                        autoFocus
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => onFieldKey(e, () => setEditing(null))}
                        onBlur={() => saveEdit(item.key)}
                      />
                    ) : (
                      <span className="text">{item.title}</span>
                    )}
                    {timing === item.key ? (
                      <span className="agenda-time-edit">
                        <input
                          inputMode="numeric"
                          maxLength={3}
                          aria-label={`Minutes for “${item.title}”`}
                          // focused with the minutes selected, so typing replaces them
                          ref={(el) => {
                            if (el && document.activeElement !== el) {
                              el.focus();
                              el.select();
                            }
                          }}
                          value={timeDraft}
                          onChange={(e) => setTimeDraft(e.target.value)}
                          onKeyDown={(e) => onFieldKey(e, () => setTiming(null))}
                          onBlur={() => saveTime(item.key)}
                        />
                        min
                      </span>
                    ) : (
                      editing !== item.key && (
                        <button
                          type="button"
                          className="agenda-time"
                          data-empty={!item.minutes}
                          onClick={() => {
                            setTiming(item.key);
                            setTimeDraft(item.minutes ? String(item.minutes) : "");
                          }}
                          aria-label={item.minutes ? `Timebox: ${item.minutes} min. Change` : `Add a timebox to “${item.title}”`}
                          title={item.minutes ? "Change the timebox" : "Add a timebox"}
                        >
                          <Icon name="clock" />
                          {item.minutes ? `${item.minutes} min` : null}
                        </button>
                      )
                    )}
                    {editing !== item.key && (
                      <button
                        type="button"
                        className="icon-btn sm"
                        onClick={() => setMenuFor(menuFor === item.key ? null : item.key)}
                        aria-haspopup="menu"
                        aria-expanded={menuFor === item.key}
                        aria-label={menuLabel}
                        disabled={rewording === item.key}
                      >
                        {rewording === item.key ? <Spinner size={16} /> : <Icon name="ellipsis" />}
                      </button>
                    )}
                    {menuFor === item.key && (
                      <div className="menu" role="menu" aria-label={menuLabel}>
                        <button
                          type="button"
                          className="menu-item"
                          role="menuitem"
                          onClick={() => {
                            setMenuFor(null);
                            setEditing(item.key);
                            setDraft(item.title);
                          }}
                        >
                          <Icon name="pencil" />
                          Edit
                        </button>
                        <button
                          type="button"
                          className="menu-item"
                          role="menuitem"
                          onClick={() => {
                            setMenuFor(null);
                            void reword(item);
                          }}
                        >
                          <Icon name="wand-sparkles" />
                          Reword with {agent}
                        </button>
                        <button
                          type="button"
                          className="menu-item"
                          role="menuitem"
                          disabled={index === 0}
                          onClick={() => {
                            setMenuFor(null);
                            move(item.key, -1);
                          }}
                        >
                          <Icon name="arrow-up" />
                          Move up
                        </button>
                        <button
                          type="button"
                          className="menu-item"
                          role="menuitem"
                          disabled={index === items.length - 1}
                          onClick={() => {
                            setMenuFor(null);
                            move(item.key, 1);
                          }}
                        >
                          <Icon name="arrow-down" />
                          Move down
                        </button>
                        <button
                          type="button"
                          className="menu-item danger"
                          role="menuitem"
                          onClick={() => {
                            setMenuFor(null);
                            save(items.filter((x) => x.key !== item.key));
                          }}
                        >
                          <Icon name="trash-2" />
                          Delete
                        </button>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
            <form onSubmit={add}>
              <div className="add-topic">
                <input
                  aria-label="Add an item"
                  value={input}
                  autoFocus={manual && items.length === 0}
                  onChange={(e) => {
                    setInput(e.target.value);
                    setRewrote(false);
                  }}
                  readOnly={rewriting}
                  placeholder="Add an item"
                />
                <button
                  type="button"
                  className="icon-btn sm wand"
                  // keep the caret in the input while the buttons are pressed
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => void rewrite()}
                  disabled={!hasText || rewriting}
                  aria-label={`Rewrite with ${agent}`}
                  title={`Rewrite with ${agent}`}
                >
                  {rewriting ? <Spinner size={16} /> : <Icon name="wand-sparkles" />}
                </button>
                <button
                  type="submit"
                  className="add"
                  onMouseDown={(e) => e.preventDefault()}
                  disabled={!hasText || rewriting}
                  aria-label="Add item"
                  title="Add item"
                >
                  <Icon name="plus" />
                </button>
              </div>
            </form>
            {rewrote && hasText && (
              <p className="rewrote" aria-live="polite">
                <Icon name="wand-sparkles" />
                Rewritten by {agent}. Press + to add it.
              </p>
            )}
          </>
        )}
        {drafted ? (
          <div className="agenda-drafted" role="status">
            <p className="agenda-drafted-head">
              <Mark size={15} />
              <span>
                {drafted.keys.length
                  ? `${agent} added ${plural(drafted.keys.length, "item")} — edit them like any other.`
                  : `${agent} found nothing new to add from open work and recent meetings.`}
              </span>
              {drafted.keys.length > 0 && (
                <button type="button" className="btn btn-quiet btn-sm" onClick={undoDraft}>
                  <Icon name="undo-2" />
                  Undo
                </button>
              )}
              <button type="button" className="icon-btn sm" onClick={() => setDrafted(null)} aria-label="Dismiss">
                <Icon name="x" />
              </button>
            </p>
            {drafted.sources.length > 0 && <p className="agenda-drafted-note">From {drafted.sources.join(", ")}</p>}
            {drafted.unavailable.length > 0 && <p className="agenda-drafted-note">Not checked: {drafted.unavailable.join("; ")}</p>}
          </div>
        ) : (
          !choosing && (
            <div className="agenda-draft">
              {draftButton}
              <span className="agenda-draft-hint">Adds items from open work and recent meetings. Yours stay as they are.</span>
            </div>
          )
        )}
        <p role="alert" className="err agenda-err">
          {error}
        </p>
      </div>
    </section>
  );
}
