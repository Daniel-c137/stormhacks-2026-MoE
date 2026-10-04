"use client";

import { type Meeting, identity, wakePhrase } from "@moe/contracts";
import { type LocalVideoTrack, Room as LiveKitRoom, createLocalVideoTrack } from "livekit-client";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Avatar } from "@/components/ui/Avatar";
import { Icon } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { Notice } from "@/components/ui/Notice";
import { useSettings } from "@/hooks/useApi";
import { useDismiss } from "@/hooks/useDismiss";
import { joinNames } from "@/lib/format";
import { LobbyAgenda } from "./LobbyAgenda";

/** What the person chose in the lobby, carried into the room. */
export interface DeviceChoices {
  micOn: boolean;
  camOn: boolean;
  micId?: string;
  camId?: string;
}

export interface LobbyProps {
  code: string;
  /** The meeting behind the code, once the team's meetings are known. */
  meeting: Meeting | null;
  /** Why the meeting behind the code could not be looked up, if it couldn't. */
  lookupError: Error | null;
  joining: boolean;
  /** Why the last attempt to join failed. */
  error: string;
  onJoin: (choices: DeviceChoices) => void;
}

function DeviceMenu({
  label,
  devices,
  selected,
  onPick,
}: {
  label: string;
  devices: MediaDeviceInfo[];
  selected: string | undefined;
  onPick: (id: string) => void;
}) {
  return (
    <div className="menu" role="menu" aria-label={label}>
      {devices.length === 0 && <p className="menu-note">No devices found. Check the browser&apos;s permission for this site.</p>}
      {devices.map((d, i) => (
        <button
          key={d.deviceId || i}
          type="button"
          className="menu-item"
          role="menuitemradio"
          aria-checked={selected ? d.deviceId === selected : i === 0}
          onClick={() => onPick(d.deviceId)}
        >
          <Icon name="check" />
          {d.label || `Device ${i + 1}`}
        </button>
      ))}
    </div>
  );
}

/** Camera preview, device pickers, transcription consent and the agenda, before joining. */
export function Lobby({ code, meeting, lookupError, joining, error, onJoin }: LobbyProps) {
  const agent = identity.agent_name;
  const { me, person } = useTeam();
  const settings = useSettings();
  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(true);
  const [micId, setMicId] = useState<string>();
  const [camId, setCamId] = useState<string>();
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [cams, setCams] = useState<MediaDeviceInfo[]>([]);
  const [camError, setCamError] = useState("");
  const [menu, setMenu] = useState<"mic" | "cam" | null>(null);
  const [infoOpen, setInfoOpen] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const controls = useRef<HTMLDivElement>(null);
  useDismiss(
    controls,
    menu !== null,
    useCallback(() => setMenu(null), []),
  );

  // Live camera preview. The track is released when the camera is turned off or the lobby closes.
  useEffect(() => {
    if (!camOn) return;
    let cancelled = false;
    let track: LocalVideoTrack | undefined;
    createLocalVideoTrack(camId ? { deviceId: camId } : undefined).then(
      (t) => {
        if (cancelled) {
          t.stop();
          return;
        }
        track = t;
        if (video.current) t.attach(video.current);
        setCamError("");
        void LiveKitRoom.getLocalDevices("videoinput", false).then((list) => {
          if (!cancelled) setCams(list);
        });
      },
      (err: unknown) => {
        if (cancelled) return;
        setCamError(err instanceof Error ? err.message : "The camera couldn't be started.");
        setCamOn(false);
      },
    );
    return () => {
      cancelled = true;
      track?.detach();
      track?.stop();
    };
  }, [camOn, camId]);

  const openMicMenu = () => {
    setMenu(menu === "mic" ? null : "mic");
    // Device names are only readable once the browser has granted microphone access.
    void LiveKitRoom.getLocalDevices("audioinput", true).then(setMics, () => setMics([]));
  };

  const others = (meeting?.participant_ids ?? []).filter((id) => id !== me.id).map(person);
  const inRoom = meeting?.status === "live" ? others : [];
  const ended = meeting && meeting.status !== "live" && meeting.status !== "scheduled";
  const isHost = meeting?.host_id === me.id;
  const hostOnly = settings.data?.who_can_allow === "host";
  const wake = settings.data?.wake_phrase || wakePhrase();
  const github = settings.data?.github;

  return (
    <main className="lobby">
      <section className="preview" data-cam={camOn} aria-label="Camera preview">
        {camOn ? (
          <>
            <div className="preview-me">
              <Avatar person={me} />
              <span className="preview-cap">camera preview</span>
            </div>
            <video ref={video} muted playsInline />
          </>
        ) : (
          <div className="preview-off">
            <Icon name="video-off" />
            <span>Your camera is off</span>
            {camError && <span className="preview-cap">{camError}</span>}
          </div>
        )}
        <div className="name-tag">{me.name}</div>

        <div className="consent">
          <div className="consent-anchor has-tip">
            <button
              type="button"
              className="consent-btn"
              onClick={() => setInfoOpen((open) => !open)}
              aria-expanded={infoOpen}
              aria-label={`${agent} transcription is on. Show details`}
            >
              <span className="mark-slot">
                <Mark size={22} tile />
              </span>
              <Icon name="scroll-text" />
            </button>
            {!infoOpen && (
              <span role="tooltip" className="tip">
                {agent} will join and transcribe this meeting.
              </span>
            )}
          </div>
          {infoOpen && (
            <div className="consent-card" role="region" aria-labelledby="h-consent">
              <header>
                <h2 id="h-consent">{agent} joins and transcribes everyone</h2>
                <button type="button" className="icon-btn sm" onClick={() => setInfoOpen(false)} aria-label="Close">
                  <Icon name="x" />
                </button>
              </header>
              <ul>
                <li>
                  <Icon name="message-circle" />
                  <span>
                    Say <b>“{wake}”</b> or use Ask {agent} to ask about past meetings, the code and Jira.
                  </span>
                </li>
                <li>
                  <Icon name="hand" />
                  <span>
                    Raises a hand when it has an answer; speaks only when {hostOnly ? "the host allows" : "someone allows"} it.
                  </span>
                </li>
                <li>
                  <Icon name="shield-check" />
                  <span>Checks technical claims against the code and flags critical mistakes.</span>
                </li>
                <li>
                  <Icon name="list-checks" />
                  <span>Drafts a summary, decisions and tasks afterwards. Nothing is sent to Jira until someone approves it.</span>
                </li>
              </ul>
              {isHost && github?.repo && (
                <div className="consent-repo">
                  <Icon name="github" />
                  <span>
                    Uses{" "}
                    <b>
                      {github.repo}
                      {github.ref ? ` @ ${github.ref}` : ""}
                    </b>
                  </span>
                  <Link className="link" href="/settings">
                    Change
                  </Link>
                </div>
              )}
              <p>By joining, you agree to be transcribed. The transcript and report are shared with your team.</p>
            </div>
          )}
        </div>

        <div className="preview-controls" ref={controls}>
          <div className="dev-split">
            <button
              type="button"
              onClick={() => setMicOn((on) => !on)}
              aria-pressed={!micOn}
              aria-label={micOn ? "Mute microphone" : "Unmute microphone"}
            >
              <Icon name={micOn ? "mic" : "mic-off"} />
              {micOn ? "Mic on" : "Muted"}
            </button>
            <span className="div" aria-hidden="true" />
            <button
              type="button"
              className="chev"
              onClick={openMicMenu}
              aria-expanded={menu === "mic"}
              aria-haspopup="menu"
              aria-label="Choose microphone"
              title="Choose microphone"
            >
              <Icon name="chevron-up" />
            </button>
            {menu === "mic" && (
              <DeviceMenu
                label="Choose microphone"
                devices={mics}
                selected={micId}
                onPick={(id) => {
                  setMicId(id);
                  setMenu(null);
                }}
              />
            )}
          </div>
          <div className="dev-split">
            <button
              type="button"
              onClick={() => setCamOn((on) => !on)}
              aria-pressed={!camOn}
              aria-label={camOn ? "Turn camera off" : "Turn camera on"}
            >
              <Icon name={camOn ? "video" : "video-off"} />
              {camOn ? "Camera on" : "Camera off"}
            </button>
            <span className="div" aria-hidden="true" />
            <button
              type="button"
              className="chev"
              onClick={() => setMenu(menu === "cam" ? null : "cam")}
              aria-expanded={menu === "cam"}
              aria-haspopup="menu"
              aria-label="Choose camera"
              title="Choose camera"
            >
              <Icon name="chevron-up" />
            </button>
            {menu === "cam" && (
              <DeviceMenu
                label="Choose camera"
                devices={cams}
                selected={camId}
                onPick={(id) => {
                  setCamId(id);
                  setCamOn(true);
                  setMenu(null);
                }}
              />
            )}
          </div>
        </div>
      </section>

      <aside className="lobby-side">
        <Link className="back-link" href="/">
          <Icon name="arrow-left" />
          Home
        </Link>
        <div className="lobby-head">
          <p className="eyebrow">{ended ? "This meeting has ended" : "Ready to join?"}</p>
          <h1 className="lobby-title">{meeting?.title ?? code}</h1>
        </div>
        {!ended && (
          <div className="already">
            {inRoom.length > 0 && (
              <div className="stack">
                {inRoom.map((p) => (
                  <Avatar key={p.id} person={p} title={p.short} />
                ))}
              </div>
            )}
            <span className="already-text">
              {inRoom.length
                ? `${joinNames(inRoom.map((p) => p.short))} ${inRoom.length === 1 ? "is" : "are"} already in.`
                : "No one's here yet."}
            </span>
          </div>
        )}
        {ended ? (
          <Link className="btn btn-primary join-big" href={`/meetings/${meeting.id}`}>
            Open the report <Icon name="arrow-right" />
          </Link>
        ) : (
          <button
            type="button"
            className="btn btn-primary join-big"
            disabled={joining}
            onClick={() => onJoin({ micOn, camOn, micId, camId })}
          >
            {joining ? "Joining…" : "Join meeting"} <Icon name="arrow-right" />
          </button>
        )}
        <span role="alert" className="err">
          {error}
        </span>
        {lookupError && !meeting && <Notice error={lookupError}>This meeting&apos;s details can&apos;t be loaded.</Notice>}
        {meeting && !ended && <LobbyAgenda meetingId={meeting.id} />}
      </aside>
    </main>
  );
}
