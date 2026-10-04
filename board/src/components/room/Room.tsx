"use client";

import {
  AGENT_PARTICIPANT_ID,
  type CodeSnippet,
  type JoinMeetingResponse,
  type Meeting,
  type Participant,
  type Person,
  type ResponseActionName,
  Topic,
  identity,
  mention,
  wakePhrase,
} from "@moe/contracts";
import { LiveKitRoom, RoomAudioRenderer, useConnectionState, useRoomContext } from "@livekit/components-react";
import { ConnectionState, DisconnectReason, type RoomOptions } from "livekit-client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Icon } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { useSettings } from "@/hooks/useApi";
import {
  HAND_ATTRIBUTE,
  useAgenda,
  useAgendaNudges,
  useAgentState,
  useChat,
  useParticipants,
  usePrivateChat,
  useResponseCards,
  useStage,
  useTranscript,
} from "@/hooks/useLive";
import { askInMeeting, describeError, endMeeting } from "@/lib/api";
import { initialsOf, shortOf } from "@/lib/format";
import { publish } from "@/lib/room";
import { AgentPresence } from "./AgentPresence";
import { CodeStage } from "./CodeStage";
import type { DeviceChoices } from "./Lobby";
import { MeetingControls } from "./MeetingControls";
import { ParticipantGrid } from "./ParticipantGrid";
import { ResponseCardView } from "./ResponseCardView";
import { type PanelMessage, SidePanel } from "./SidePanel";

export interface RoomProps {
  join: JoinMeetingResponse;
  choices: DeviceChoices;
}

/** How long a speaker keeps the stage after they stop, so the layout doesn't jump between sentences. */
const SPEAKER_HOLD_MS = 3000;
/** How long a caption stays up after its last update. */
const CAPTION_HOLD_MS = 6000;
/** How long to wait for the agent to acknowledge a button press or a chat mention. */
const AGENT_WAIT_MS = 30_000;

/** Errors from the browser's getUserMedia/getDisplayMedia: the room is fine, a device is not. */
const DEVICE_ERRORS = new Set(["NotAllowedError", "NotFoundError", "NotReadableError", "OverconstrainedError", "SecurityError"]);
const isDeviceError = (err: Error) => DEVICE_ERRORS.has(err.name);

function useStickySpeaker(participants: Participant[]): string | null {
  const talking = participants.filter((p) => !p.is_agent && p.is_speaking && p.mic_on).map((p) => p.id);
  const [held, setHeld] = useState<string | null>(null);
  // Whoever already has the stage keeps it while they are still talking.
  const current = held && talking.includes(held) ? held : (talking[0] ?? null);
  useEffect(() => {
    if (current) {
      setHeld(current);
      return;
    }
    const timer = setTimeout(() => setHeld(null), SPEAKER_HOLD_MS);
    return () => clearTimeout(timer);
  }, [current]);
  return current ?? held;
}

/** LiveKit room: participant grid or code stage, agent presence, chat panel, controls. */
export function Room({ join, choices }: RoomProps) {
  const router = useRouter();
  const ended = useRef(false);
  const wasConnected = useRef(false);
  const [connectError, setConnectError] = useState("");
  const [deviceError, setDeviceError] = useState("");
  const options = useMemo<RoomOptions>(
    () => ({
      adaptiveStream: true,
      dynacast: true,
      audioCaptureDefaults: choices.micId ? { deviceId: choices.micId } : undefined,
      videoCaptureDefaults: choices.camId ? { deviceId: choices.camId } : undefined,
    }),
    [choices.micId, choices.camId],
  );
  const report = `/meetings/${join.meeting.id}`;

  // Stable handlers: LiveKitRoom reconnects whenever one of them changes identity.
  const onConnected = useCallback(() => {
    wasConnected.current = true;
  }, []);
  // LiveKitRoom reports a microphone or camera that can't start through onError too; that is not a
  // failed connection, so it is told as a device problem while the meeting carries on.
  const onError = useCallback((err: Error) => {
    if (isDeviceError(err)) setDeviceError(`Your microphone or camera couldn't start: ${err.message}`);
    else setConnectError(err.message);
  }, []);
  const onMediaDeviceFailure = useCallback(() => setDeviceError("Your microphone or camera couldn't start."), []);
  const onDisconnected = useCallback(
    (reason?: DisconnectReason) => {
      // Never got in: stay here and say why, rather than bouncing home without a word.
      if (!wasConnected.current) {
        setConnectError((message) => message || "The meeting server couldn't be reached.");
        return;
      }
      // The host ending the meeting closes the room for everyone; that leads to the report.
      router.push(ended.current || reason === DisconnectReason.ROOM_DELETED ? report : "/");
    },
    [router, report],
  );
  const onEnded = useCallback((value: boolean) => {
    ended.current = value;
  }, []);

  return (
    <LiveKitRoom
      serverUrl={join.livekit_url}
      token={join.token}
      connect
      audio={choices.micOn}
      video={choices.camOn}
      options={options}
      onConnected={onConnected}
      onError={onError}
      onMediaDeviceFailure={onMediaDeviceFailure}
      onDisconnected={onDisconnected}
      style={{ display: "contents" }}
    >
      <RoomAudioRenderer />
      <RoomView meeting={join.meeting} connectError={connectError} deviceError={deviceError} onEnded={onEnded} />
    </LiveKitRoom>
  );
}

function RoomView({
  meeting,
  connectError,
  deviceError,
  onEnded,
}: {
  meeting: Meeting;
  connectError: string;
  deviceError: string;
  onEnded: (ended: boolean) => void;
}) {
  const agent = identity.agent_name;
  const room = useRoomContext();
  const connection = useConnectionState();
  const { me, members } = useTeam();
  const settings = useSettings();

  const participants = useParticipants(meeting.host_id);
  const agentState = useAgentState();
  const transcript = useTranscript();
  const cards = useResponseCards();
  const publicChat = useChat(meeting.id);
  const privateChat = usePrivateChat();
  const { stage, setStage } = useStage();
  const agenda = useAgenda(meeting.id);
  const { nudges, dismiss: dismissNudge } = useAgendaNudges(meeting.id);

  const [chatOpen, setChatOpen] = useState(true);
  const [captionsOn, setCaptionsOn] = useState(false);
  const [captionFresh, setCaptionFresh] = useState(false);
  const [ending, setEnding] = useState(false);
  const [toast, setToast] = useState("");
  const [copied, setCopied] = useState(false);
  const [askPending, setAskPending] = useState(false);
  const [busyCard, setBusyCard] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState<string[]>([]);
  const [agentThread, setAgentThread] = useState<PanelMessage[]>([]);
  const [privateAsks, setPrivateAsks] = useState(0);
  const [mentionedAt, setMentionedAt] = useState<number | null>(null);
  const [chatError, setChatError] = useState("");

  const isHost = meeting.host_id === me.id;
  const humans = participants.filter((p) => !p.is_agent);
  const agentPresent = participants.some((p) => p.is_agent);
  const handUp = participants.find((p) => p.id === me.id)?.hand_raised ?? false;
  const featuredId = useStickySpeaker(participants);
  const wake = settings.data?.wake_phrase || wakePhrase();
  const canAct = settings.data?.who_can_allow === "host" ? isHost : true;

  const personOf = useCallback(
    (p: Participant): Person =>
      p.id === me.id
        ? me
        : (members.find((m) => m.id === p.id) ?? { id: p.id, name: p.name, short: shortOf(p.name), initials: initialsOf(p.name) }),
    [me, members],
  );

  useEffect(() => {
    if (deviceError) setToast(deviceError);
  }, [deviceError]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 6000);
    return () => clearTimeout(timer);
  }, [toast]);

  // ---- ask the agent by voice: the button arms it, the worker takes the next thing you say
  useEffect(() => {
    if (!askPending) return;
    if (agentState.state !== "idle") {
      setAskPending(false);
      return;
    }
    const timer = setTimeout(() => {
      setAskPending(false);
      setToast(`${agent} didn't respond. Try again, or say “${wake}”.`);
    }, 8000);
    return () => clearTimeout(timer);
  }, [askPending, agentState.state, agent, wake]);

  const signalAsk = (cancel: boolean) => {
    setAskPending(!cancel);
    publish(room, Topic.ASK, { by_id: me.id, cancel }, [AGENT_PARTICIPANT_ID]).catch((err: unknown) => {
      setAskPending(false);
      setToast(describeError(err));
    });
  };

  // ---- the shared answer card
  const card = [...cards].reverse().find((c) => c.status === "pending" && !dismissed.includes(c.id)) ?? null;
  const cardKey = card ? `${card.id}:${card.status}` : "";
  useEffect(() => {
    setBusyCard(null);
  }, [cardKey]);
  useEffect(() => {
    if (!busyCard) return;
    const timer = setTimeout(() => setBusyCard(null), 5000);
    return () => clearTimeout(timer);
  }, [busyCard]);

  const onCardAction = (action: ResponseActionName) => {
    if (!card) return;
    if (action === "show_on_stage") {
      const snippet = card.answer.snippets[0];
      if (snippet) setStage(snippet.id).catch((err: unknown) => setToast(describeError(err)));
    } else {
      setBusyCard(card.id);
      if (action === "dismiss") setDismissed((ids) => [...ids, card.id]);
    }
    publish(room, Topic.RESPONSE_ACTION, { card_id: card.id, action, by_id: me.id }, [AGENT_PARTICIPANT_ID]).catch(
      (err: unknown) => setToast(describeError(err)),
    );
  };

  // ---- code on stage: any snippet an answer in this meeting has carried
  const snippets = useMemo(() => {
    const map = new Map<string, CodeSnippet>();
    for (const c of cards) for (const s of c.answer.snippets) map.set(s.id, s);
    for (const m of agentThread) if (m.snippet) map.set(m.snippet.id, m.snippet);
    return map;
  }, [cards, agentThread]);
  const staged = stage.snippet_id ? (snippets.get(stage.snippet_id) ?? null) : null;

  // ---- captions: the segment being spoken now, or the last one finished
  const caption = [...transcript].reverse().find((s) => !s.is_final) ?? transcript[transcript.length - 1] ?? null;
  const captionKey = caption ? `${caption.seg_id}:${caption.text.length}` : "";
  useEffect(() => {
    if (!captionKey) return;
    setCaptionFresh(true);
    const timer = setTimeout(() => setCaptionFresh(false), CAPTION_HOLD_MS);
    return () => clearTimeout(timer);
  }, [captionKey]);
  const captionWho = caption
    ? caption.speaker_id === AGENT_PARTICIPANT_ID
      ? agent
      : shortOf((members.find((m) => m.id === caption.speaker_id) ?? (caption.speaker_id === me.id ? me : null))?.name ?? caption.speaker_name)
    : "";
  const captionWords = caption ? caption.text.split(" ") : [];
  const captionText = captionWords.length > 20 ? `…${captionWords.slice(-20).join(" ")}` : captionWords.join(" ");

  // ---- chat: public room chat, private messages (the agent's fact-checks among them), and the
  // private thread with the agent
  const messages = useMemo(() => {
    const fromRoom = publicChat.messages.map((m): PanelMessage => {
      // An answer the agent posted to chat carries its card's sources and code with it.
      const source = m.is_agent ? cards.find((c) => c.answer.text === m.text) : undefined;
      return source ? { ...m, sources: source.answer.sources, snippet: source.answer.snippets[0] ?? null } : m;
    });
    return [...fromRoom, ...privateChat.messages, ...agentThread].sort((a, b) => a.ts.localeCompare(b.ts));
  }, [publicChat.messages, privateChat.messages, agentThread, cards]);

  const lastAgentPublicAt = publicChat.messages.reduce((t, m) => (m.is_agent ? Math.max(t, Date.parse(m.ts)) : t), 0);
  const awaitingMention = mentionedAt !== null && lastAgentPublicAt < mentionedAt;
  useEffect(() => {
    if (mentionedAt === null) return;
    const timer = setTimeout(() => setMentionedAt(null), AGENT_WAIT_MS);
    return () => clearTimeout(timer);
  }, [mentionedAt]);

  const askPrivately = async (text: string) => {
    const base = { meeting_id: meeting.id, visibility: "private" as const };
    // Earlier turns of this private thread go with the question, so a follow-up has its context.
    const history = agentThread.map((m) => ({ role: m.is_agent ? ("agent" as const) : ("user" as const), text: m.text }));
    setAgentThread((list) => [
      ...list,
      {
        ...base,
        id: crypto.randomUUID(),
        sender_id: me.id,
        sender_name: me.name,
        is_agent: false,
        text,
        ts: new Date().toISOString(),
        recipient_id: AGENT_PARTICIPANT_ID,
      },
    ]);
    setPrivateAsks((n) => n + 1);
    try {
      const answer = await askInMeeting(meeting.id, { question: text, visibility: "private", history });
      setAgentThread((list) => [
        ...list,
        {
          ...base,
          id: answer.id,
          sender_id: AGENT_PARTICIPANT_ID,
          sender_name: agent,
          is_agent: true,
          text: answer.text,
          ts: new Date().toISOString(),
          recipient_id: me.id,
          sources: answer.sources,
          unavailable: answer.unavailable,
          snippet: answer.snippets[0] ?? null,
        },
      ]);
    } catch (err) {
      setChatError(`${agent} couldn't answer. ${describeError(err)}`);
    } finally {
      setPrivateAsks((n) => n - 1);
    }
  };

  const onSend = (text: string, to: string | null) => {
    setChatError("");
    const failed = (err: unknown) => setChatError(`The message wasn't sent. ${describeError(err)}`);
    if (to === AGENT_PARTICIPANT_ID) {
      void askPrivately(text);
    } else if (to) {
      privateChat
        .send({
          id: crypto.randomUUID(),
          meeting_id: meeting.id,
          sender_id: me.id,
          sender_name: me.name,
          is_agent: false,
          text,
          ts: new Date().toISOString(),
          visibility: "private",
          recipient_id: to,
        })
        .catch(failed);
    } else {
      publicChat.send(text).then(() => {
        if (text.toLowerCase().includes(mention().toLowerCase())) setMentionedAt(Date.now());
      }, failed);
    }
  };

  // ---- controls
  const toggleHand = () => {
    room.localParticipant
      .setAttributes({ [HAND_ATTRIBUTE]: handUp ? "" : "1" })
      .catch(() => setToast("Your hand couldn't be raised: this meeting doesn't allow it."));
  };

  const end = async () => {
    setEnding(true);
    onEnded(true);
    try {
      await endMeeting(meeting.id);
      await room.disconnect();
    } catch (err) {
      onEnded(false);
      setEnding(false);
      setToast(`The meeting wasn't ended. ${describeError(err)}`);
    }
  };

  const copyLink = () => {
    navigator.clipboard.writeText(`${window.location.origin}/m/${meeting.code}`).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      },
      () => setToast("The link couldn't be copied."),
    );
  };

  const connected = connection === ConnectionState.Connected;

  return (
    <div className="room">
      <div className="fx-main">
        <header className="fx-top">
          <div className="fx-pill fx-id">
            <span className="mark-slot">
              <Mark size={26} />
            </span>
            <span className="fx-divider" aria-hidden="true" />
            <h1 className="room-name">{meeting.title}</h1>
            <button type="button" className="code-chip" onClick={copyLink} aria-label={`Meeting code ${meeting.code}. Copy the link`} title="Copy meeting link">
              {copied ? "Link copied" : meeting.code}
              <Icon name={copied ? "check" : "copy"} />
            </button>
          </div>
          <div className="fx-pill fx-atlas">
            {card ? (
              <ResponseCardView card={card} canAct={canAct} busy={busyCard === card.id} onAction={onCardAction} />
            ) : (
              <AgentPresence
                state={agentState}
                askPending={askPending}
                agentPresent={agentPresent}
                wake={wake}
                onAsk={() => signalAsk(false)}
                onCancel={() => signalAsk(true)}
              />
            )}
            <span className="sr" aria-live="polite">
              {card ? `${agent} has an answer ready.` : agentState.detail}
            </span>
            <span
              className="scribe"
              role="img"
              data-off={!agentPresent}
              aria-label={agentPresent ? `${agent} is listening and transcribing` : `${agent} is not in this meeting; nothing is being transcribed`}
              title={agentPresent ? `${agent} is listening and transcribing` : `${agent} is not in this meeting; nothing is being transcribed`}
            >
              <Icon name="scroll-text" />
            </span>
          </div>
        </header>

        <section className="stage" aria-label="Stage">
          {connectError ? (
            <p className="stage-note">
              <span>
                Couldn&apos;t connect to the meeting. {connectError}{" "}
                <Link className="link" href="/">
                  Back to meetings
                </Link>
              </span>
            </p>
          ) : !connected ? (
            <p className="stage-note">{connection === ConnectionState.Reconnecting ? "Reconnecting…" : "Connecting…"}</p>
          ) : staged ? (
            <CodeStage snippet={staged} onClose={() => void setStage(null).catch(() => undefined)} />
          ) : (
            <ParticipantGrid participants={humans} meId={me.id} featuredId={featuredId} personOf={personOf} />
          )}
          {captionsOn && (
            <div className="captions" role="region" aria-label="Live captions" aria-live="polite">
              {caption && captionFresh ? (
                <p>
                  <b>{captionWho}:</b> {captionText}
                </p>
              ) : (
                !transcript.length && <p>{agentPresent ? "Captions are on." : `Captions need ${agent} in the meeting.`}</p>
              )}
            </div>
          )}
        </section>

        {nudges.length > 0 && (
          <div className="room-notes">
            {nudges.map((n) => (
              <div key={n.key} className="nudge" role="status">
                <Icon name="clock" />
                <span>{n.text}</span>
                <button type="button" className="icon-btn sm" onClick={() => dismissNudge(n.key)} aria-label="Dismiss this reminder">
                  <Icon name="x" />
                </button>
              </div>
            ))}
          </div>
        )}
        {toast && (
          <p className="room-toast" role="status">
            {toast}
          </p>
        )}
        <MeetingControls
          isHost={isHost}
          handUp={handUp}
          captionsOn={captionsOn}
          chatOpen={chatOpen}
          ending={ending}
          onToggleHand={toggleHand}
          onToggleCaptions={() => setCaptionsOn((on) => !on)}
          onToggleChat={() => setChatOpen((open) => !open)}
          onLeave={() => void room.disconnect()}
          onEnd={() => void end()}
          onError={setToast}
        />
      </div>
      {chatOpen && (
        <SidePanel
          messages={messages}
          me={me}
          people={humans.filter((p) => p.id !== me.id)}
          personOf={personOf}
          agentTyping={privateAsks > 0 || awaitingMention}
          error={chatError}
          agenda={agenda.agenda}
          agendaError={agenda.error}
          onSend={onSend}
          onClose={() => setChatOpen(false)}
        />
      )}
    </div>
  );
}
