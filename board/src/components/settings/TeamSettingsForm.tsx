"use client";

import { type Sensitivity, type TeamSettings, type Voice, identity } from "@moe/contracts";
import { type ChangeEvent, type FormEvent, useEffect, useRef, useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { TeamAccounts } from "@/components/settings/TeamAccounts";
import { Avatar } from "@/components/ui/Avatar";
import { Icon, Spinner } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";
import { Connectors } from "./Connectors";
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

/** Profile, account, connectors, the agent's voice and fact-check sensitivity. */
export function TeamSettingsForm({ onClose }: { onClose: () => void }) {
  const agent = identity.agent_name;
  const { me, email, setMe } = useTeam();
  const admin = me.is_admin === true; // only an admin changes team settings and creates accounts
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
  // Connectors save on their own (PUT /settings/connectors); the latest saved choice shows here.
  const [savedConnectors, setSavedConnectors] = useState<TeamSettings | null>(null);
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

  // Plays the voice's sample, or stops it when it is already playing.
  const previewVoice = (voiceId: string, sample: string) => {
    audio.current?.pause();
    if (previewing === voiceId) {
      setPreviewing(null);
      return;
    }
    setPreviewing(voiceId);
    // A voice's sample is either a clip to play or the line it would say.
    if (/^https?:\/\//.test(sample)) {
      const clip = new Audio(sample);
      clip.onended = () => setPreviewing((now) => (now === voiceId ? null : now));
      audio.current = clip;
      void clip.play().catch(() => setProblem("The voice sample couldn't be played."));
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

          {admin && <TeamAccounts />}

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
              {!admin && <p className="note admin-only">Only admins can change these.</p>}
              <div className="sgroup" role="group" aria-labelledby="g-conn">
                <h2 id="g-conn">Connectors</h2>
                <Connectors
                  settings={savedConnectors ?? settings.data ?? s}
                  statuses={connectors}
                  canEdit={admin}
                  onSaved={(saved) => {
                    setSavedConnectors(saved);
                    settings.reload();
                  }}
                />
              </div>

              <fieldset className="sgroup" disabled={!admin} aria-labelledby="g-agent">
                <h2 id="g-agent">{agent}</h2>
                <section className="srow" aria-labelledby="s-voice">
                  <h3 id="s-voice">Voice</h3>
                  {voices.data ? (
                    voices.data.length === 0 ? (
                      <p className="note">No voices are available.</p>
                    ) : (
                      <VoicePicker
                        voices={voices.data}
                        saved={s.voice}
                        previewing={previewing}
                        onChange={(voice) => {
                          audio.current?.pause();
                          setPreviewing(null);
                          set({ voice });
                        }}
                        onPreview={previewVoice}
                      />
                    )
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
              </fieldset>
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

/** The agent's voice as one select (names only) with a play/stop preview of the chosen one,
 * instead of a card per voice: an ElevenLabs account can list hundreds (#137). No saved choice
 * means the agent's default voice; a saved voice the account no longer has stays selected,
 * marked unavailable, rather than silently becoming another. */
function VoicePicker({
  voices,
  saved,
  previewing,
  onChange,
  onPreview,
}: {
  voices: Voice[];
  saved: string | null | undefined;
  previewing: string | null;
  onChange: (voice: string) => void;
  onPreview: (voiceId: string, sample: string) => void;
}) {
  const fallback = voices.find((v) => v.default_label) ?? voices[0];
  const chosen = saved ?? fallback?.id ?? "";
  const current = voices.find((v) => v.id === chosen);
  const playing = current !== undefined && previewing === current.id;
  return (
    <div className="voice-pick">
      <div className="voice-row">
        <select className="field" aria-labelledby="s-voice" value={chosen} onChange={(e) => onChange(e.target.value)}>
          {saved && !current && <option value={saved}>Saved voice (no longer available)</option>}
          {voices.map((v) => (
            <option key={v.id} value={v.id}>
              {v.default_label ? `${v.name} (default)` : v.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn btn-outline voice-play"
          onClick={() => current && onPreview(current.id, current.sample)}
          disabled={!current?.sample}
          aria-pressed={playing}
          aria-label={current ? `${playing ? "Stop" : "Preview"} ${current.name}` : "Preview"}
        >
          <Icon name={playing ? "square" : "play"} />
          {playing ? "Stop" : "Preview"}
        </button>
      </div>
      {playing && current && !/^https?:\/\//.test(current.sample) && <p className="note">“{current.sample}”</p>}
    </div>
  );
}
