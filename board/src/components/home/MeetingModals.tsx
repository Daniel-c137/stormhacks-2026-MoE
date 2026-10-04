"use client";

import { type Meeting, identity } from "@moe/contracts";
import { useRouter } from "next/navigation";
import { type FormEvent, type KeyboardEvent, type ReactNode, useEffect, useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Avatar } from "@/components/ui/Avatar";
import { Icon, Spinner } from "@/components/ui/Icon";
import { draftAgendaInto } from "@/lib/agenda";
import { createMeeting, describeError } from "@/lib/api";
import { fmtClock, fmtDate, fmtDuration, isoDay, parseMeetingCode } from "@/lib/format";

const DURATIONS = [15, 30, 45, 60, 90, 120];

function Modal({ titleId, onClose, children }: { titleId: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape" && !e.defaultPrevented) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby={titleId} onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}

function ModalHead({ id, title, onClose }: { id: string; title: string; onClose: () => void }) {
  return (
    <div className="modal-head">
      <h2 id={id} className="modal-title">
        {title}
      </h2>
      <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
        <Icon name="x" />
      </button>
    </div>
  );
}

/** Start a meeting now and go to its lobby, or schedule one for later with invitees. */
export function NewMeetingModal({ onClose, onScheduled }: { onClose: () => void; onScheduled: (meeting: Meeting) => void }) {
  const router = useRouter();
  const { me, members, membersError } = useTeam();
  const [title, setTitle] = useState("");
  const [when, setWhen] = useState<"now" | "later">("now");
  const [date, setDate] = useState(() => isoDay(new Date(Date.now() + 86_400_000)));
  const [time, setTime] = useState("10:00");
  const [duration, setDuration] = useState(30);
  const [invitees, setInvitees] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [inviteOpen, setInviteOpen] = useState(false);
  const [draftAgenda, setDraftAgenda] = useState(false);
  const [busy, setBusy] = useState<"" | "creating" | "drafting">("");
  // Created, but the agenda draft failed: the meeting stands and the dialog only says so.
  const [scheduled, setScheduled] = useState<Meeting | null>(null);
  const [error, setError] = useState("");
  const agent = identity.agent_name;

  const start = new Date(`${date}T${time || "10:00"}`);
  const validStart = !Number.isNaN(start.getTime());
  const range = validStart
    ? `${fmtDate(start)}, ${fmtClock(start)}–${fmtClock(new Date(start.getTime() + duration * 60_000))} (${fmtDuration(duration)})`
    : "Pick a date and start time";

  const q = query.trim().toLowerCase();
  const results = members.filter(
    (p) => p.id !== me.id && !invitees.includes(p.id) && (!q || `${p.name} ${p.title ?? ""}`.toLowerCase().includes(q)),
  );
  const add = (id: string) => {
    setInvitees((xs) => [...xs, id]);
    setQuery("");
  };
  const onInviteKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (results[0]) add(results[0].id);
    }
    if (e.key === "Escape") {
      // Close the picker, not the whole dialog.
      e.preventDefault();
      e.currentTarget.blur();
    }
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (when === "later" && !validStart) {
      setError("Pick a date and start time.");
      return;
    }
    if (scheduled) {
      onScheduled(scheduled);
      return;
    }
    setBusy("creating");
    setError("");
    let meeting: Meeting;
    try {
      const name = title.trim() || "Untitled meeting";
      if (when === "now") {
        meeting = await createMeeting({ title: name });
        router.push(`/m/${meeting.code}`);
        return;
      }
      meeting = await createMeeting({
        title: name,
        scheduled_start: start.toISOString(),
        duration_min: duration,
        invitee_ids: invitees,
      });
    } catch (err) {
      setError(`The meeting wasn't created. ${describeError(err)}`);
      setBusy("");
      return;
    }
    if (draftAgenda) {
      setBusy("drafting");
      try {
        await draftAgendaInto(meeting.id);
      } catch (err) {
        setScheduled(meeting);
        setError(`The meeting is scheduled, but ${agent} couldn't draft its agenda. ${describeError(err)} You can draft it from the lobby.`);
        setBusy("");
        return;
      }
    }
    onScheduled(meeting);
  };

  return (
    <Modal titleId="new-meeting-title" onClose={onClose}>
      <form onSubmit={submit}>
        <ModalHead id="new-meeting-title" title="New meeting" onClose={onClose} />
        <input
          className="field"
          aria-label="Title"
          autoFocus
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Title"
          style={{ height: 52, fontSize: 17, fontWeight: 600 }}
        />
        <div className="seg" role="radiogroup" aria-label="When">
          <button type="button" role="radio" aria-checked={when === "now"} onClick={() => setWhen("now")}>
            <Icon name="zap" />
            Start now
          </button>
          <button type="button" role="radio" aria-checked={when === "later"} onClick={() => setWhen("later")}>
            <Icon name="calendar" />
            Schedule for later
          </button>
        </div>
        {when === "later" && (
          <>
            <div className="sched">
              <div className="sched-grid">
                <label className="label">
                  Date
                  <input className="field" type="date" value={date} min={isoDay(new Date())} onChange={(e) => setDate(e.target.value)} />
                </label>
                <label className="label">
                  Start
                  <input className="field" type="time" value={time} step={900} onChange={(e) => setTime(e.target.value)} />
                </label>
                <label className="label">
                  Duration
                  <select className="field" value={duration} onChange={(e) => setDuration(Number(e.target.value))}>
                    {DURATIONS.map((m) => (
                      <option key={m} value={m}>
                        {fmtDuration(m)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <span className="sched-range">
                <Icon name="clock" />
                {range}
              </span>
            </div>
            <label className="check-row">
              <input type="checkbox" checked={draftAgenda} onChange={(e) => setDraftAgenda(e.target.checked)} disabled={Boolean(scheduled)} />
              <span className="person-text">
                <span className="person-name">Let {agent} draft the agenda</span>
                <span className="person-meta">From open work and recent meetings, once the meeting is created. Edit it in the lobby.</span>
              </span>
            </label>
            <div className="sched">
              <span id="inv-h" className="label">
                Invitees
              </span>
              <ul className="people-list" aria-labelledby="inv-h">
                <li className="person-row">
                  <Avatar person={me} me />
                  <span className="person-text">
                    <span className="person-name">{me.name} (you)</span>
                    <span className="person-meta">Organiser</span>
                  </span>
                </li>
                {invitees.map((id) => {
                  const p = members.find((m) => m.id === id);
                  if (!p) return null;
                  return (
                    <li key={id} className="person-row">
                      <Avatar person={p} />
                      <span className="person-text">
                        <span className="person-name">{p.name}</span>
                        {p.title && <span className="person-meta">{p.title}</span>}
                      </span>
                      <button
                        type="button"
                        className="icon-btn sm"
                        onClick={() => setInvitees((xs) => xs.filter((x) => x !== id))}
                        aria-label={`Remove ${p.name}`}
                      >
                        <Icon name="x" />
                      </button>
                    </li>
                  );
                })}
              </ul>
              <label className="invite-field">
                <Icon name={inviteOpen ? "search" : "user-plus"} />
                <input
                  className="field"
                  role="combobox"
                  aria-expanded={inviteOpen}
                  aria-controls="inv-results"
                  aria-label="Add people"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={onInviteKey}
                  onFocus={() => setInviteOpen(true)}
                  onBlur={() => {
                    setInviteOpen(false);
                    setQuery("");
                  }}
                  placeholder="Add people"
                />
              </label>
              {inviteOpen && (
                <>
                  <ul id="inv-results" className="results" role="listbox" aria-label={`People on ${identity.product_name}`}>
                    {results.map((p) => (
                      <li key={p.id}>
                        <button
                          type="button"
                          className="result"
                          role="option"
                          aria-selected={false}
                          // mousedown, so the pick lands before the input's blur closes the list
                          onMouseDown={(e) => {
                            e.preventDefault();
                            add(p.id);
                          }}
                        >
                          <Icon name="plus" />
                          <span className="result-name">{p.name}</span>
                          {p.title && <span className="result-meta">{p.title}</span>}
                        </button>
                      </li>
                    ))}
                  </ul>
                  {results.length === 0 && (
                    <p className="person-meta" style={{ paddingLeft: 14 }}>
                      {membersError
                        ? `Your team can't be loaded. ${describeError(membersError)}`
                        : q
                          ? `No one on your team matches “${query.trim()}”.`
                          : "Everyone on your team has been added."}
                    </p>
                  )}
                </>
              )}
            </div>
          </>
        )}
        <span role="alert" className="err">
          {error}
        </span>
        <div className="modal-foot">
          {scheduled ? (
            <button type="submit" className="btn btn-primary">
              Done
            </button>
          ) : (
            <>
              <button type="button" className="btn btn-outline" onClick={onClose}>
                Cancel
              </button>
              <button type="submit" className="btn btn-primary" disabled={Boolean(busy)}>
                {busy === "drafting" ? "Drafting the agenda…" : when === "later" ? "Create" : "Join"}
                {busy ? <Spinner /> : <Icon name={when === "later" ? "calendar-check" : "arrow-right"} />}
              </button>
            </>
          )}
        </div>
      </form>
    </Modal>
  );
}

/** Join by code or pasted link. Checked against the team's meetings when that list is known. */
export function JoinMeetingModal({ meetings, onClose }: { meetings: Meeting[] | undefined; onClose: () => void }) {
  const router = useRouter();
  const [text, setText] = useState("");
  const [error, setError] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const code = parseMeetingCode(text);
    if (!code) {
      setError("Enter a meeting code or link.");
      return;
    }
    const match = meetings?.find((m) => m.code.toLowerCase() === code.toLowerCase());
    if (meetings && (!match || (match.status !== "live" && match.status !== "scheduled"))) {
      setError("No live meeting with that code.");
      return;
    }
    router.push(`/m/${encodeURIComponent(match?.code ?? code)}`);
  };

  return (
    <Modal titleId="join-meeting-title" onClose={onClose}>
      <form onSubmit={submit}>
        <ModalHead id="join-meeting-title" title="Join meeting" onClose={onClose} />
        <input
          className="field mono"
          aria-label="Code or link"
          autoFocus
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setError("");
          }}
          placeholder="Code or link"
          style={{ height: 52, fontSize: 16 }}
        />
        <span role="alert" className="err">
          {error}
        </span>
        <div className="modal-foot">
          <button type="button" className="btn btn-outline" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary">
            <Icon name="log-in" />
            Join
          </button>
        </div>
      </form>
    </Modal>
  );
}
