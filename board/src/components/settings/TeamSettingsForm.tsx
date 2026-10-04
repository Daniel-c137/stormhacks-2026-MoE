"use client";

import { type ConnectorName, type ConnectorStatus, type Sensitivity, type TeamSettings, identity } from "@moe/contracts";
import { type ChangeEvent, type FormEvent, useEffect, useRef, useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Avatar } from "@/components/ui/Avatar";
import { Icon, Spinner } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { Notice } from "@/components/ui/Notice";
import { useConnectors, useSettings, useVoices } from "@/hooks/useApi";
import { ApiError, changePassword, deletePhoto, describeError, updateMe, updateSettings, uploadPhoto, waitText } from "@/lib/api";
import { signOut } from "@/lib/auth";
import { initialsOf } from "@/lib/format";

const SENSITIVITY: [Sensitivity, string, string][] = [
  [
    "quiet",
    "Critical only",
    "Raise a hand only when a claim is wrong and acting on it would cause a bug, outage or security issue. Everything else is logged silently.",
  ],
  ["balanced", "Balanced", "Also raise a hand when a wrong claim would mislead a decision. Minor slips are logged silently."],
  ["eager", "Eager", "Raise a hand for any claim that looks wrong, including minor slips."],
];

const PHOTO_SIZE = 256;
const PREVIEW_MS = 4000;

/** A new photo picked here, not saved yet: the upload and its local preview. */
interface PickedPhoto {
  blob: Blob;
  preview: string;
}

const MIN_PASSWORD = 10;

/** Centre-crop to a small square JPEG, so a profile photo stays a few kilobytes. */
async function squarePhoto(file: File): Promise<Blob> {
  const bitmap = await createImageBitmap(file);
  const side = Math.min(bitmap.width, bitmap.height);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = PHOTO_SIZE;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser can't resize images.");
  context.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, PHOTO_SIZE, PHOTO_SIZE);
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("The image couldn't be encoded."))), "image/jpeg", 0.85),
  );
}

const CONNECTOR_TEXT: Record<ConnectorStatus["state"], string> = {
  connected: "Connected",
  not_configured: "Not configured",
  failing: "Not reachable",
};

/** Whether the server can use an integration right now, checked live (GET /settings/connectors). */
function ConnectorState({ name, statuses }: { name: ConnectorName; statuses: ReturnType<typeof useConnectors> }) {
  const status = statuses.data?.find((c) => c.name === name);
  if (!status) {
    return (
      <div className="connected" data-on={false}>
        <Icon name="circle-dashed" />
        {statuses.error ? `Status unavailable. ${describeError(statuses.error)}` : "Checking…"}
      </div>
    );
  }
  return (
    <div className="connected" data-on={status.state === "connected"}>
      <Icon name={status.state === "connected" ? "circle-check" : status.state === "failing" ? "triangle-alert" : "circle-dashed"} />
      {CONNECTOR_TEXT[status.state]}
      {status.detail && <span>· {status.detail}</span>}
    </div>
  );
}

/** POST /auth/password. Sessions already issued stay valid until they expire. */
function PasswordForm() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (next.length < MIN_PASSWORD) {
      setMessage({ ok: false, text: `The new password must be at least ${MIN_PASSWORD} characters.` });
      return;
    }
    if (next !== again) {
      setMessage({ ok: false, text: "The new passwords don't match." });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      await changePassword({ current_password: current, new_password: next });
      setCurrent("");
      setNext("");
      setAgain("");
      setMessage({ ok: true, text: "Password changed." });
    } catch (err) {
      const text =
        err instanceof ApiError && err.status === 401
          ? "The current password is wrong."
          : err instanceof ApiError && err.status === 429
            ? `Too many failed attempts.${err.retryAfter ? ` Try again in ${waitText(err.retryAfter)}.` : ""}`
            : `The password wasn't changed. ${describeError(err)}`;
      setMessage({ ok: false, text });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="srow-stack" onSubmit={submit} aria-labelledby="s-password">
      <label className="label" style={{ maxWidth: 360 }}>
        Current password
        <input
          className="field"
          type="password"
          autoComplete="current-password"
          value={current}
          onChange={(e) => setCurrent(e.target.value)}
          required
        />
      </label>
      <div className="two">
        <label className="label">
          New password
          <input
            className="field"
            type="password"
            autoComplete="new-password"
            value={next}
            onChange={(e) => setNext(e.target.value)}
            minLength={MIN_PASSWORD}
            required
          />
        </label>
        <label className="label">
          Repeat new password
          <input
            className="field"
            type="password"
            autoComplete="new-password"
            value={again}
            onChange={(e) => setAgain(e.target.value)}
            minLength={MIN_PASSWORD}
            required
          />
        </label>
      </div>
      <div className="photo-acts">
        <div>
          <button type="submit" className="btn btn-outline btn-sm" disabled={busy || !current || !next || !again}>
            {busy ? <Spinner /> : <Icon name="lock" />}
            Change password
          </button>
        </div>
        <span role="status" className={message && !message.ok ? "err" : "note"}>
          {message?.text ?? `At least ${MIN_PASSWORD} characters.`}
        </span>
      </div>
    </form>
  );
}

/** Profile, GitHub repo and ref, Jira site and project, the agent's voice and fact-check sensitivity. */
export function TeamSettingsForm({ onClose }: { onClose: () => void }) {
  const agent = identity.agent_name;
  const { me, email, setMe } = useTeam();
  const settings = useSettings();
  const voices = useVoices();
  const connectors = useConnectors();

  // Each is null/undefined until the person changes it; the saved value shows until then.
  const [name, setName] = useState<string | null>(null);
  // undefined: unchanged; null: remove the saved photo.
  const [photo, setPhoto] = useState<PickedPhoto | null | undefined>(undefined);
  const [draft, setDraft] = useState<TeamSettings | null>(null);
  const [previewing, setPreviewing] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [problem, setProblem] = useState("");
  const audio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    if (!previewing) return;
    const timer = setTimeout(() => setPreviewing(null), PREVIEW_MS);
    return () => clearTimeout(timer);
  }, [previewing]);
  useEffect(() => () => audio.current?.pause(), []);
  const preview = photo?.preview;
  useEffect(() => (preview ? () => URL.revokeObjectURL(preview) : undefined), [preview]);

  const s = draft ?? settings.data ?? null;
  const shownName = name ?? me.name;
  const shownPhoto = photo === undefined ? (me.photo_url ?? null) : (photo?.preview ?? null);
  const profileChanged = (name !== null && name.trim() !== me.name) || photo !== undefined;
  const touch = () => {
    setSaved(false);
    setProblem("");
  };
  const set = (patch: Partial<TeamSettings>) => {
    if (!s) return;
    setDraft({ ...s, ...patch });
    touch();
  };

  const onPhoto = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const blob = await squarePhoto(file);
      setPhoto({ blob, preview: URL.createObjectURL(blob) });
      touch();
    } catch (err) {
      setProblem(`That image couldn't be used. ${describeError(err)}`);
    }
  };

  const previewVoice = (voiceId: string, sample: string) => {
    setPreviewing(voiceId);
    audio.current?.pause();
    // A voice's sample is either a clip to play or the line it would say.
    if (/^https?:\/\//.test(sample)) {
      audio.current = new Audio(sample);
      void audio.current.play().catch(() => setProblem("The voice sample couldn't be played."));
    }
  };

  const save = async () => {
    setSaving(true);
    setProblem("");
    try {
      if (name !== null && name.trim() !== me.name) {
        setMe(await updateMe({ name: name.trim() }));
        setName(null);
      }
      if (photo !== undefined) {
        setMe(await (photo ? uploadPhoto(photo.blob) : deletePhoto()));
        setPhoto(undefined);
      }
      if (draft) {
        await updateSettings(draft);
        settings.reload();
        connectors.reload();
        setDraft(null);
      }
      setSaved(true);
    } catch (err) {
      setProblem(`Your changes aren't saved. ${describeError(err)}`);
    } finally {
      setSaving(false);
    }
  };

  const options = SENSITIVITY.filter(([value]) => value !== "eager" || s?.sensitivity === "eager");
  const minutes = s?.interrupt_minutes ?? 0;
  const indexed = s?.github.indexed_at ? new Date(s.github.indexed_at).toLocaleString() : null;
  const githubStatus = [s?.github.files != null && `${s.github.files.toLocaleString()} files indexed`, indexed && `at ${indexed}`]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="scrim" onClick={onClose}>
      <div className="settings" role="dialog" aria-modal="true" aria-labelledby="h-settings" onClick={(e) => e.stopPropagation()}>
        <div className="settings-head">
          <h1 id="h-settings">Settings</h1>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close settings">
            <Icon name="x" />
          </button>
        </div>
        <div className="settings-body">
          <div className="sgroup" role="group" aria-labelledby="g-profile">
            <h2 id="g-profile">Profile</h2>
            <section className="srow" aria-labelledby="s-photo">
              <h3 id="s-photo">Photo</h3>
              <div className="photo-row">
                <Avatar person={{ name: shownName, initials: initialsOf(shownName), photo_url: shownPhoto }} me />
                <div className="photo-acts">
                  <div>
                    <label className="btn btn-outline btn-sm file-btn">
                      <Icon name="image-up" />
                      {shownPhoto ? "Change photo" : "Upload photo"}
                      <input type="file" accept="image/jpeg,image/png" onChange={(e) => void onPhoto(e)} />
                    </label>
                    {shownPhoto && (
                      <button
                        type="button"
                        className="btn btn-danger-quiet btn-sm"
                        onClick={() => {
                          setPhoto(null);
                          touch();
                        }}
                      >
                        <Icon name="trash-2" />
                        Remove
                      </button>
                    )}
                  </div>
                  <span className="note">Shown on your tile when your camera is off. JPG or PNG.</span>
                </div>
              </div>
            </section>
            <section className="srow" aria-labelledby="s-name">
              <h3 id="s-name">Display name</h3>
              <div className="field-note">
                <input
                  className="field"
                  aria-labelledby="s-name"
                  value={shownName}
                  onChange={(e) => {
                    setName(e.target.value);
                    touch();
                  }}
                  style={{ maxWidth: 360 }}
                />
                <span className="note">How you appear in meetings, transcripts and reports.</span>
              </div>
            </section>
          </div>

          <div className="sgroup" role="group" aria-labelledby="g-account">
            <h2 id="g-account">Account</h2>
            <section className="srow" aria-labelledby="s-password">
              <h3 id="s-password">Password</h3>
              <PasswordForm />
            </section>
            <section className="srow" aria-labelledby="s-signout">
              <h3 id="s-signout">Session</h3>
              <div className="photo-acts">
                <div>
                  <button type="button" className="btn btn-outline btn-sm" onClick={signOut}>
                    <Icon name="log-out" />
                    Sign out
                  </button>
                </div>
                {email && <span className="note">Signed in as {email}.</span>}
              </div>
            </section>
          </div>

          {!s && (
            <div className="sgroup" style={{ padding: 22 }}>
              {settings.error ? (
                <Notice error={settings.error} onRetry={settings.reload}>
                  Team settings can&apos;t be loaded.
                </Notice>
              ) : (
                <p className="muted-p">Loading team settings…</p>
              )}
            </div>
          )}

          {s && (
            <>
              <div className="sgroup" role="group" aria-labelledby="g-conn">
                <h2 id="g-conn">Connectors</h2>
                <section className="srow" aria-labelledby="s-code">
                  <h3 id="s-code">Codebase</h3>
                  <div>
                    <div className="two">
                      <label className="label">
                        GitHub repository
                        <input
                          className="field mono"
                          value={s.github.repo ?? ""}
                          onChange={(e) => set({ github: { ...s.github, repo: e.target.value || null } })}
                          placeholder="owner/repo"
                        />
                      </label>
                      <label className="label">
                        Branch or tag
                        <input
                          className="field mono"
                          value={s.github.ref ?? ""}
                          onChange={(e) => set({ github: { ...s.github, ref: e.target.value || null } })}
                          placeholder="main"
                        />
                      </label>
                    </div>
                    <ConnectorState name="github" statuses={connectors} />
                    {githubStatus && <span className="note">{githubStatus}</span>}
                  </div>
                </section>
                <section className="srow" aria-labelledby="s-jira">
                  <h3 id="s-jira">Jira</h3>
                  <div>
                    <div className="two">
                      <label className="label">
                        Site
                        <input
                          className="field"
                          value={s.jira.site ?? ""}
                          onChange={(e) => set({ jira: { ...s.jira, site: e.target.value || null } })}
                          placeholder="your-team.atlassian.net"
                        />
                      </label>
                      <label className="label">
                        Project key
                        <input
                          className="field mono"
                          value={s.jira.project ?? ""}
                          onChange={(e) => set({ jira: { ...s.jira, project: e.target.value.toUpperCase() || null } })}
                          style={{ textTransform: "uppercase" }}
                        />
                      </label>
                    </div>
                    <ConnectorState name="jira" statuses={connectors} />
                  </div>
                </section>
              </div>

              <div className="sgroup" role="group" aria-labelledby="g-agent">
                <h2 id="g-agent">{agent}</h2>
                <section className="srow" aria-labelledby="s-voice">
                  <h3 id="s-voice">{agent} voice</h3>
                  {voices.data ? (
                    <div className="voices" role="radiogroup" aria-labelledby="s-voice">
                      {voices.data.length === 0 && <p className="note">No voices are available.</p>}
                      {voices.data.map((v) => (
                        <div key={v.id} className="voice">
                          <label className="opt-label">
                            <input type="radio" name="voice" checked={s.voice === v.id} onChange={() => set({ voice: v.id })} />
                            <span>
                              <span className="opt-name">{v.name}</span>
                              <span className="opt-desc">{v.desc}</span>
                            </span>
                          </label>
                          <div className="voice-foot">
                            <button
                              type="button"
                              className="btn btn-outline btn-sm"
                              onClick={() => previewVoice(v.id, v.sample)}
                              aria-label={`Preview ${v.name}`}
                            >
                              <Icon name="play" />
                              Preview
                            </button>
                            {previewing === v.id && (
                              <span className="playing">
                                <span className="mark-slot">
                                  <Mark size={20} tile state="speaking" />
                                </span>
                                <em>{/^https?:\/\//.test(v.sample) ? "Playing…" : `“${v.sample}”`}</em>
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : voices.error ? (
                    <Notice error={voices.error} onRetry={voices.reload}>
                      Voices can&apos;t be loaded.
                    </Notice>
                  ) : (
                    <p className="note">Loading voices…</p>
                  )}
                </section>
                <section className="srow" aria-labelledby="s-fc">
                  <h3 id="s-fc">Fact-checking</h3>
                  <div className="srow-stack">
                    <div className="sens" role="radiogroup" aria-label="Sensitivity">
                      {options.map(([value, label, desc]) => (
                        <label key={value} className="opt-label">
                          <input type="radio" name="sens" checked={s.sensitivity === value} onChange={() => set({ sensitivity: value })} />
                          <span>
                            <span className="opt-name">{label}</span>
                            <span className="opt-desc">{desc}</span>
                          </span>
                        </label>
                      ))}
                    </div>
                    <div>
                      <label htmlFor="s-int" className="opt-name">
                        Interruption limit
                      </label>
                      <div className="range-row">
                        <input
                          id="s-int"
                          type="range"
                          min={1}
                          max={15}
                          step={1}
                          value={minutes}
                          onChange={(e) => set({ interrupt_minutes: Number(e.target.value) })}
                        />
                        <span className="range-val">{minutes} min</span>
                      </div>
                      <p className="note">
                        At most one unprompted hand every {minutes} minute{minutes === 1 ? "" : "s"}. Answers to direct
                        questions don&apos;t count.
                      </p>
                    </div>
                  </div>
                </section>
              </div>
            </>
          )}
        </div>
        <div className="settings-foot">
          <span role="status" className={problem ? "err" : "note"}>
            {problem || (saved ? "Settings saved." : "")}
          </span>
          <button type="button" className="btn btn-primary" onClick={() => void save()} disabled={saving || (!draft && !profileChanged)}>
            {saving ? <Spinner /> : <Icon name={saved ? "check" : "save"} />}
            {saved ? "Saved" : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}
