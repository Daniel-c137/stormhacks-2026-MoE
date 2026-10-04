"use client";

import { type Agenda, type AgendaItemInput, identity } from "@moe/contracts";
import { type FormEvent, type KeyboardEvent, useCallback, useRef, useState } from "react";
import { Icon, Spinner } from "@/components/ui/Icon";
import { useAgenda } from "@/hooks/useApi";
import { useDismiss } from "@/hooks/useDismiss";
import { describeError, rewriteAgendaItem, updateAgenda } from "@/lib/api";

/** A topic as edited here. `key` is stable for React; `id` is the brain's, once it has saved it. */
interface Topic extends AgendaItemInput {
  key: string;
}

const fromAgenda = (agenda: Agenda): Topic[] =>
  agenda.items.map((item) => ({ key: item.id, id: item.id, title: item.title, minutes: item.minutes ?? null }));

/** Topics for the meeting: add, edit, delete, and have the agent tidy the wording. Saved with the
 * meeting (PUT /meetings/{id}/agenda takes the whole list; items without an id are new). */
export function LobbyAgenda({ meetingId }: { meetingId: string }) {
  const agent = identity.agent_name;
  const agenda = useAgenda(meetingId);
  // Edits made here; until the first one, the saved agenda is shown as it is.
  const [edited, setEdited] = useState<Topic[] | null>(null);
  const items = edited ?? (agenda.data ? fromAgenda(agenda.data) : []);
  const [input, setInput] = useState("");
  const [rewriting, setRewriting] = useState(false);
  const [rewrote, setRewrote] = useState(false);
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
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

  const save = (next: Topic[]) => {
    setEdited(next);
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

  const hasText = Boolean(input.trim());

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

  const saveEdit = (id: string) => {
    if (editing !== id) return;
    const title = draft.trim();
    if (title) save(items.map((x) => (x.key === id ? { ...x, title } : x)));
    setEditing(null);
  };

  const onEditKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      e.currentTarget.blur();
    }
    if (e.key === "Escape") {
      e.preventDefault();
      setEditing(null);
    }
  };

  return (
    <section className="agenda" aria-labelledby="h-agenda">
      <h2 id="h-agenda">What do you want to talk about?</h2>
      <div className="agenda-scroll">
        <ul className="agenda-list" ref={list}>
          {items.map((item) => {
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
                    onKeyDown={onEditKey}
                    onBlur={() => saveEdit(item.key)}
                  />
                ) : (
                  <>
                    <span className="text">{item.title}</span>
                    <button
                      type="button"
                      className="icon-btn sm"
                      onClick={() => setMenuFor(menuFor === item.key ? null : item.key)}
                      aria-haspopup="menu"
                      aria-expanded={menuFor === item.key}
                      aria-label={menuLabel}
                    >
                      <Icon name="ellipsis" />
                    </button>
                  </>
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
              aria-label="Add a topic"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                setRewrote(false);
              }}
              readOnly={rewriting}
              placeholder="Add a topic"
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
              aria-label="Add topic"
              title="Add topic"
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
        <p role="alert" className="err agenda-err">
          {error}
        </p>
      </div>
    </section>
  );
}
