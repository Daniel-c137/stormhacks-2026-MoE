// Live meeting state from the LiveKit room and its data topics. Use inside the room only.
import type {
  Agenda,
  AgentState,
  ChatMessage,
  FactCheck,
  Participant,
  ResponseCard,
  StagePayload,
  TranscriptSegment,
} from "@moe/contracts";
import { notImplemented } from "@/lib/stub";

export const useParticipants = (): Participant[] => notImplemented("useParticipants");
export const useAgentState = (): AgentState => notImplemented("useAgentState");
/** Partial and final segments; partials are replaced in place by seg_id. */
export const useTranscript = (): TranscriptSegment[] => notImplemented("useTranscript");
export const useChat = (): ChatMessage[] => notImplemented("useChat");
export const useResponseCards = (): ResponseCard[] => notImplemented("useResponseCards");
/** Public fact-checks plus private ones addressed to this participant. */
export const useFactChecks = (): FactCheck[] => notImplemented("useFactChecks");
export const useAgenda = (): Agenda | null => notImplemented("useAgenda");
export const useStage = (): StagePayload => notImplemented("useStage");
