"use client";

import { useLocalParticipant } from "@livekit/components-react";
import { useCallback, useRef, useState } from "react";
import { Icon, type IconName, Spinner } from "@/components/ui/Icon";
import { useDismiss } from "@/hooks/useDismiss";

export interface MeetingControlsProps {
  isHost: boolean;
  handUp: boolean;
  captionsOn: boolean;
  chatOpen: boolean;
  /** The host pressed End and the request is in flight. */
  ending: boolean;
  onToggleHand: () => void;
  onToggleCaptions: () => void;
  onToggleChat: () => void;
  onLeave: () => void;
  onEnd: () => void;
  /** A device or permission problem worth telling the person about. */
  onError: (message: string) => void;
}

interface Control {
  label: string;
  icon: IconName;
  pressed: boolean;
  /** Pressed because something is switched off (muted, camera off), drawn as a warning. */
  off?: boolean;
  onClick: () => void;
}

/** Mic, camera, screen share, hand, captions, chat and leave; end meeting for the host. */
export function MeetingControls(props: MeetingControlsProps) {
  const { localParticipant, isMicrophoneEnabled: mic, isCameraEnabled: cam, isScreenShareEnabled: sharing } = useLocalParticipant();
  const [leaveOpen, setLeaveOpen] = useState(false);
  const leaveWrap = useRef<HTMLDivElement>(null);
  useDismiss(
    leaveWrap,
    leaveOpen,
    useCallback(() => setLeaveOpen(false), []),
  );

  const attempt = (action: Promise<unknown>, what: string) =>
    action.catch((err: unknown) => props.onError(`${what} ${err instanceof Error ? err.message : ""}`.trim()));

  const controls: Control[] = [
    {
      label: mic ? "Mute microphone" : "Unmute microphone",
      icon: mic ? "mic" : "mic-off",
      pressed: !mic,
      off: true,
      onClick: () => void attempt(localParticipant.setMicrophoneEnabled(!mic), "The microphone couldn't be switched."),
    },
    {
      label: cam ? "Turn camera off" : "Turn camera on",
      icon: cam ? "video" : "video-off",
      pressed: !cam,
      off: true,
      onClick: () => void attempt(localParticipant.setCameraEnabled(!cam), "The camera couldn't be switched."),
    },
    {
      label: sharing ? "Stop sharing" : "Share screen",
      icon: sharing ? "monitor-x" : "monitor-up",
      pressed: sharing,
      onClick: () => void attempt(localParticipant.setScreenShareEnabled(!sharing), "Screen sharing didn't start."),
    },
    { label: props.handUp ? "Lower hand" : "Raise hand", icon: "hand", pressed: props.handUp, onClick: props.onToggleHand },
    {
      label: props.captionsOn ? "Turn off captions" : "Turn on captions",
      icon: "captions",
      pressed: props.captionsOn,
      onClick: props.onToggleCaptions,
    },
  ];

  return (
    <footer className="fx-dock">
      <div className="fx-dock-group">
        {controls.map((c) => (
          <button
            key={c.icon}
            type="button"
            className="ctl"
            onClick={c.onClick}
            aria-pressed={c.pressed}
            aria-label={c.label}
            title={c.label}
            data-off={c.off}
          >
            <Icon name={c.icon} />
          </button>
        ))}
      </div>
      <span className="fx-dock-sep" aria-hidden="true" />
      <div className="fx-dock-group">
        <button
          type="button"
          className="pbtn"
          onClick={props.onToggleChat}
          aria-pressed={props.chatOpen}
          aria-label={props.chatOpen ? "Hide chat" : "Show chat"}
          title={props.chatOpen ? "Hide chat" : "Show chat"}
        >
          <Icon name="message-square" />
        </button>
      </div>
      <div className="leave-wrap" ref={leaveWrap}>
        <button
          type="button"
          className="leave"
          onClick={props.isHost ? () => setLeaveOpen((open) => !open) : props.onLeave}
          aria-haspopup={props.isHost ? "menu" : undefined}
          aria-expanded={props.isHost ? leaveOpen : undefined}
        >
          {props.ending ? <Spinner /> : <Icon name="phone-off" />}
          <span>Leave</span>
        </button>
        {leaveOpen && (
          <div className="menu" role="menu" aria-label="Leave or end the meeting">
            <button type="button" className="menu-item" role="menuitem" onClick={props.onLeave}>
              <Icon name="phone-off" />
              Leave meeting
            </button>
            <button
              type="button"
              className="menu-item danger"
              role="menuitem"
              disabled={props.ending}
              onClick={() => {
                setLeaveOpen(false);
                props.onEnd();
              }}
            >
              <Icon name="x" />
              End meeting for everyone
            </button>
          </div>
        )}
      </div>
    </footer>
  );
}
