"use client";

// Live meeting state from the LiveKit room and its data topics. Use inside the room only, and call
// each hook once (in Room): they collect events from the moment they mount.
import {
  AGENT_PARTICIPANT_ID,
  type Agenda,
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
import { getAgenda } from "@/lib/api";
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

/** The meeting's agenda: the saved one, loaded once on joining so a late joiner sees it, then each
 * update the agent publishes as it keeps time. A newer revision always wins over an older one. */
export function useAgenda(meetingId: string): { agenda: Agenda | null; error: Error | null } {
  const [agenda, setAgenda] = useState<Agenda | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const newer = (next: Agenda) => (current: Agenda | null) =>
    current && (current.revision ?? 0) > (next.revision ?? 0) ? current : next;
  useEffect(() => {
    let cancelled = false;
    getAgenda(meetingId).then(
      (saved) => {
        if (cancelled) return;
        setError(null);
        // Only fill in or move forward: a live update may have arrived while this was loading.
        setAgenda((current) => (current && (current.revision ?? 0) >= (saved.revision ?? 0) ? current : saved));
      },
      (err: Error) => {
        if (!cancelled) setError(err);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [meetingId]);
  useTopic(Topic.AGENDA, (next) => {
    if (next.meeting_id !== meetingId) return;
    setError(null);
    setAgenda(newer(next));
  });
  return { agenda, error };
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
