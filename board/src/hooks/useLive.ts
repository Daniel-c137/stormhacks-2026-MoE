"use client";

// Live meeting state from the LiveKit room and its data topics. Use inside the room only, and call
// each hook once (in Room): they collect events from the moment they mount.
import {
  AGENT_PARTICIPANT_ID,
  type Agenda,
  type AgendaItem,
  type AgendaItemStatus,
  type AgendaNudge,
  type AgentState,
  type ChatMessage,
  type Participant,
  type ResponseCard,
  type StagePayload,
  Topic,
  type TopicPayloads,
  type TranscriptSegment,
  identity,
} from "@moe/contracts";
import { useChat as useLiveKitChat, useRoomContext } from "@livekit/components-react";
import { type RemoteParticipant, RoomEvent } from "livekit-client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getAgenda, updateAgenda } from "@/lib/api";
import { publish } from "@/lib/room";

/** Set on a participant's LiveKit attributes while their hand is up. */
export const HAND_ATTRIBUTE = "hand_raised";

const decoder = new TextDecoder();

/**
 * Subscribe to one data topic. Agent topics are only accepted from the agent's own identity, so a
 * participant cannot forge a caption, card or agent state.
 */
function useTopic<T extends Topic>(
  topic: T,
  onPayload: (payload: TopicPayloads[T], from: RemoteParticipant | undefined) => void,
  agentOnly = true,
) {
  const room = useRoomContext();
  const handler = useRef(onPayload);
  useEffect(() => {
    handler.current = onPayload;
  });
  useEffect(() => {
    const onData = (bytes: Uint8Array, from?: RemoteParticipant, _kind?: unknown, dataTopic?: string) => {
      if (dataTopic !== topic) return;
      if (agentOnly && from?.identity !== AGENT_PARTICIPANT_ID) return;
      let payload: TopicPayloads[T];
      try {
        payload = JSON.parse(decoder.decode(bytes));
      } catch {
        return;
      }
      handler.current(payload, from);
    };
    room.on(RoomEvent.DataReceived, onData);
    return () => {
      room.off(RoomEvent.DataReceived, onData);
    };
  }, [room, topic, agentOnly]);
}

const PARTICIPANT_EVENTS = [
  RoomEvent.Connected,
  RoomEvent.Reconnected,
  RoomEvent.ParticipantConnected,
  RoomEvent.ParticipantDisconnected,
  RoomEvent.ParticipantNameChanged,
  RoomEvent.ParticipantAttributesChanged,
  RoomEvent.ActiveSpeakersChanged,
  RoomEvent.TrackPublished,
  RoomEvent.TrackUnpublished,
  RoomEvent.TrackSubscribed,
  RoomEvent.TrackUnsubscribed,
  RoomEvent.TrackMuted,
  RoomEvent.TrackUnmuted,
  RoomEvent.LocalTrackPublished,
  RoomEvent.LocalTrackUnpublished,
] as const;

/** Everyone in the room, the local participant first. id is the account id and the LiveKit identity. */
export function useParticipants(hostId: string): Participant[] {
  const room = useRoomContext();
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const bump = () => setVersion((v) => v + 1);
    for (const event of PARTICIPANT_EVENTS) room.on(event, bump);
    return () => {
      for (const event of PARTICIPANT_EVENTS) room.off(event, bump);
    };
  }, [room]);
  return useMemo(
    () =>
      [room.localParticipant, ...room.remoteParticipants.values()].map((p) => ({
        id: p.identity,
        name: p.name || p.identity,
        role: p.identity === hostId ? ("host" as const) : ("member" as const),
        is_agent: p.identity === AGENT_PARTICIPANT_ID,
        mic_on: p.isMicrophoneEnabled,
        cam_on: p.isCameraEnabled,
        is_speaking: p.isSpeaking,
        hand_raised: p.attributes[HAND_ATTRIBUTE] === "1",
      })),
    // version is the change signal; the room object itself is mutated in place
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [room, hostId, version],
  );
}

const IDLE: AgentState = { state: "idle", detail: "", hand_urgency: "normal", hand_reason: "" };

export function useAgentState(): AgentState {
  const [state, setState] = useState<AgentState>(IDLE);
  useTopic(Topic.AGENT_STATE, setState);
  return state;
}

const MAX_SEGMENTS = 200;

/** Partial and final segments; partials are replaced in place by seg_id. */
export function useTranscript(): TranscriptSegment[] {
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  useTopic(Topic.TRANSCRIPT, (seg) =>
    setSegments((list) => {
      const i = list.findIndex((x) => x.seg_id === seg.seg_id);
      if (i >= 0) return list.map((x, j) => (j === i ? seg : x));
      return [...list, seg].slice(-MAX_SEGMENTS);
    }),
  );
  return segments;
}

/** Public room chat (LiveKit's built-in chat), saved with the meeting by the worker. */
export function useChat(meetingId: string): { messages: ChatMessage[]; send: (text: string) => Promise<void> } {
  const { chatMessages, send } = useLiveKitChat();
  const messages = useMemo(
    () =>
      chatMessages.map(
        (m): ChatMessage => ({
          id: m.id,
          meeting_id: meetingId,
          sender_id: m.from?.identity ?? "",
          sender_name: m.from?.name || m.from?.identity || "Someone",
          is_agent: m.from?.identity === AGENT_PARTICIPANT_ID,
          text: m.message,
          ts: new Date(m.timestamp).toISOString(),
          visibility: "public",
        }),
      ),
    [chatMessages, meetingId],
  );
  const sendText = useCallback(
    async (text: string) => {
      await send(text);
    },
    [send],
  );
  return { messages, send: sendText };
}

/** Private messages to this participant: from another person, or a fact-check from the agent about
 * something they said. Sent to one participant only; never stored anywhere. */
export function usePrivateChat(): { messages: ChatMessage[]; send: (message: ChatMessage) => Promise<void> } {
  const room = useRoomContext();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  useTopic(
    Topic.PRIVATE_CHAT,
    (message, from) => {
      // The sender is whoever LiveKit says sent it, not what the payload claims: a message is the
      // agent's only when it comes from the agent's own identity.
      if (!from || message.recipient_id !== room.localParticipant.identity) return;
      const fromAgent = from.identity === AGENT_PARTICIPANT_ID;
      setMessages((list) => [
        ...list,
        {
          ...message,
          sender_id: from.identity,
          sender_name: fromAgent ? identity.agent_name : from.name || from.identity,
          is_agent: fromAgent,
          visibility: "private",
        },
      ]);
    },
    false,
  );
  const send = useCallback(
    async (message: ChatMessage) => {
      if (!message.recipient_id) return;
      await publish(room, Topic.PRIVATE_CHAT, message, [message.recipient_id]);
      setMessages((list) => [...list, message]);
    },
    [room],
  );
  return { messages, send };
}

/** Shared answers to voice questions, newest last. A card is replaced when its status changes. */
export function useResponseCards(): ResponseCard[] {
  const [cards, setCards] = useState<ResponseCard[]>([]);
  useTopic(Topic.RESPONSE_CARD, (card) =>
    setCards((list) => (list.some((c) => c.id === card.id) ? list.map((c) => (c.id === card.id ? card : c)) : [...list, card])),
  );
  return cards;
}

/** How long a "covered" notice stays up, and how long its slide-out takes. */
const NOTICE_MS = 8000;
const NOTICE_OUT_MS = 260;

/** The agent just marked an agenda item covered. */
export interface CoveredNotice {
  key: string;
  item_id: string;
  title: string;
  /** Sliding out; removed when the animation ends. */
  leaving: boolean;
}

/** A person's own tick that the brain has not confirmed yet. */
type Tick = Pick<AgendaItem, "status" | "covered_by" | "covered_t">;

export interface LiveAgenda {
  agenda: Agenda | null;
  error: Error | null;
  /** Tick an item as covered, untick it, or skip it. Rejects if the brain refuses. */
  setStatus: (itemId: string, status: AgendaItemStatus) => Promise<void>;
  /** Items the agent has just covered, newest last. */
  notices: CoveredNotice[];
  dismissNotice: (key: string) => void;
}

/**
 * The meeting's agenda: the saved one, loaded once on joining so a late joiner sees it, then each
 * update the agent publishes as it keeps time. A newer revision always wins over an older one.
 *
 * A person's tick is saved through the brain and shown at once; everyone else gets it with the
 * agent's next update. When an update shows the agent covered an item, a notice is raised.
 */
export function useAgenda(meetingId: string, meId: string, startedAt: string | null | undefined): LiveAgenda {
  const [agenda, setAgenda] = useState<Agenda | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [ticks, setTicks] = useState<Record<string, Tick>>({});
  const [notices, setNotices] = useState<CoveredNotice[]>([]);
  // The newest agenda seen, readable inside handlers without waiting for a render.
  const latest = useRef<Agenda | null>(null);
  const timers = useRef(new Set<ReturnType<typeof setTimeout>>());

  const later = useCallback((ms: number, run: () => void) => {
    const timer = setTimeout(() => {
      timers.current.delete(timer);
      run();
    }, ms);
    timers.current.add(timer);
  }, []);
  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const timer of pending) clearTimeout(timer);
    };
  }, []);

  /** Take `next` unless what is held is already newer (or as new, when `unlessSame`). */
  const take = useCallback((next: Agenda, unlessSame = false) => {
    const held = latest.current;
    const heldRev = held?.revision ?? 0;
    const nextRev = next.revision ?? 0;
    if (held && (unlessSame ? heldRev >= nextRev : heldRev > nextRev)) return;
    latest.current = next;
    setAgenda(next);
  }, []);

  useEffect(() => {
    let cancelled = false;
    getAgenda(meetingId).then(
      (saved) => {
        if (cancelled) return;
        setError(null);
        // Only fill in or move forward: a live update may have arrived while this was loading.
        take(saved, true);
      },
      (err: Error) => {
        if (!cancelled) setError(err);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [meetingId, take]);

  const dismissNotice = useCallback(
    (key: string) => {
      setNotices((list) => list.map((n) => (n.key === key ? { ...n, leaving: true } : n)));
      later(NOTICE_OUT_MS, () => setNotices((list) => list.filter((n) => n.key !== key)));
    },
    [later],
  );

  useTopic(Topic.AGENDA, (next) => {
    if (next.meeting_id !== meetingId) return;
    setError(null);
    const held = latest.current;
    // Only a change seen happening counts: an item this client knew as not covered, now covered
    // by the agent. Joining a meeting where items are already covered raises nothing.
    if (held && (held.revision ?? 0) < (next.revision ?? 0)) {
      const was = new Map(held.items.map((i) => [i.id, i.status]));
      for (const item of next.items) {
        const before = was.get(item.id);
        if (item.status !== "covered" || item.covered_by !== AGENT_PARTICIPANT_ID) continue;
        if (before === undefined || before === "covered") continue;
        const key = `${item.id}:${crypto.randomUUID()}`;
        setNotices((list) => [...list.filter((n) => n.item_id !== item.id), { key, item_id: item.id, title: item.title, leaving: false }]);
        later(NOTICE_MS, () => dismissNotice(key));
      }
    }
    take(next);
  });

  const setStatus = useCallback(
    async (itemId: string, status: AgendaItemStatus) => {
      const held = latest.current;
      if (!held) return;
      const covered = status === "covered";
      const elapsed = startedAt ? Math.max(0, (Date.now() - Date.parse(startedAt)) / 1000) : null;
      setTicks((all) => ({
        ...all,
        [itemId]: { status, covered_by: covered ? meId : null, covered_t: covered ? elapsed : null },
      }));
      try {
        // The whole list goes back, as the lobby sends it; only this item's status is named, so
        // every other item keeps the status the brain has for it.
        const saved = await updateAgenda(meetingId, {
          items: held.items.map((i) => ({ id: i.id, title: i.title, minutes: i.minutes ?? null, status: i.id === itemId ? status : null })),
        });
        take(saved);
      } finally {
        setTicks(({ [itemId]: _done, ...rest }) => rest);
      }
    },
    [meetingId, meId, startedAt, take],
  );

  const shown = useMemo(
    () => (agenda ? { ...agenda, items: agenda.items.map((i) => (ticks[i.id] ? { ...i, ...ticks[i.id] } : i)) } : null),
    [agenda, ticks],
  );
  // A notice about an item that is no longer covered (someone undid it) has nothing left to say;
  // one already sliding out finishes its exit.
  const current = useMemo(
    () => notices.filter((n) => n.leaving || shown?.items.some((i) => i.id === n.item_id && i.status === "covered")),
    [notices, shown],
  );
  return { agenda: shown, error, setStatus, notices: current, dismissNotice };
}

/** How long a timebox nudge stays up. */
const NUDGE_MS = 12_000;

export interface ShownNudge extends AgendaNudge {
  key: string;
}

/** Timebox nudges from the agent, each shown for a while and then dropped. Never spoken. */
export function useAgendaNudges(meetingId: string): { nudges: ShownNudge[]; dismiss: (key: string) => void } {
  const [nudges, setNudges] = useState<ShownNudge[]>([]);
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const dismiss = useCallback((key: string) => {
    clearTimeout(timers.current.get(key));
    timers.current.delete(key);
    setNudges((list) => list.filter((n) => n.key !== key));
  }, []);
  useTopic(Topic.AGENDA_NUDGE, (nudge) => {
    if (nudge.meeting_id !== meetingId) return;
    const key = `${nudge.item_id}:${crypto.randomUUID()}`;
    setNudges((list) => [...list.filter((n) => n.item_id !== nudge.item_id), { ...nudge, key }]);
    timers.current.set(key, setTimeout(() => dismiss(key), NUDGE_MS));
  });
  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const timer of pending.values()) clearTimeout(timer);
    };
  }, []);
  return { nudges, dismiss };
}

/** The snippet everyone is looking at, and a way to put one on stage or clear it. */
export function useStage(): { stage: StagePayload; setStage: (snippetId: string | null) => Promise<void> } {
  const room = useRoomContext();
  const [stage, setLocal] = useState<StagePayload>({ snippet_id: null });
  useTopic(Topic.STAGE, setLocal, false);
  const setStage = useCallback(
    async (snippetId: string | null) => {
      const payload = { snippet_id: snippetId };
      setLocal(payload);
      await publish(room, Topic.STAGE, payload);
    },
    [room],
  );
  return { stage, setStage };
}
