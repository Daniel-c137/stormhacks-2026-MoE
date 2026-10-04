"use client";

import { type AgendaItem, identity } from "@moe/contracts";
import { type FormEvent, type KeyboardEvent, useCallback, useRef, useState } from "react";
import { Icon, Spinner } from "@/components/ui/Icon";
import { useAgenda } from "@/hooks/useApi";
import { useDismiss } from "@/hooks/useDismiss";
import { describeError, rewriteTopic, updateAgenda } from "@/lib/api";

/** Topics for the meeting: add, edit, delete, and have the agent tidy the wording. Saved with the meeting. */
export function LobbyAgenda({ meetingId }: { meetingId: string }) {
  const agent = identity.agent_name;
  const agenda = useAgenda(meetingId);
  // Edits made here; until the first one, the saved agenda is shown as it is.
  const [edited, setEdited] = useState<AgendaItem[] | null>(null);
  const items = edited ?? agenda.data?.items ?? [];
  const [input, setInput] = useState("");
  const [rewriting, setRewriting] = useState(false);
  const [rewrote, setRewrote] = useState(false);
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const list = useRef<HTMLUListElement>(null);
  useDismiss(
    list,
    menuFor !== null,
    useCallback(() => setMenuFor(null), []),
  );

  const save = (next: AgendaItem[]) => {
    setEdited(next);
    setError("");
    updateAgenda(meetingId, { items: next }).catch((err: unknown) =>
      setError(`These topics aren't saved yet. ${describeError(err)}`),
    );
  };

  const hasText = Boolean(input.trim());

  const add = (e: FormEvent) => {
    e.preventDefault();
    const title = input.trim();
    if (!title || rewriting) return;
    save([...items, { id: crypto.randomUUID(), title, sources: [], status: "pending" }]);
    setInput("");
    setRewrote(false);
  };

  const rewrite = async () => {
    const text = input.trim();
    if (!text) return;
    setRewriting(true);
    setError("");
    try {
      const result = await rewriteTopic(meetingId, { text });
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
    if (title) save(items.map((x) => (x.id === id ? { ...x, title } : x)));
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
              <li key={item.id} className="agenda-item">
                {editing === item.id ? (
                  <input
                    className="field"
                    aria-label="Edit topic"
                    autoFocus
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={onEditKey}
                    onBlur={() => saveEdit(item.id)}
                  />
                ) : (
                  <>
                    <span className="text">{item.title}</span>
                    <button
                      type="button"
                      className="icon-btn sm"
                      onClick={() => setMenuFor(menuFor === item.id ? null : item.id)}
                      aria-haspopup="menu"
                      aria-expanded={menuFor === item.id}
                      aria-label={menuLabel}
                    >
                      <Icon name="ellipsis" />
                    </button>
                  </>
                )}
                {menuFor === item.id && (
                  <div className="menu" role="menu" aria-label={menuLabel}>
                    <button
                      type="button"
                      className="menu-item"
                      role="menuitem"
                      onClick={() => {
                        setMenuFor(null);
                        setEditing(item.id);
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
                        save(items.filter((x) => x.id !== item.id));
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
