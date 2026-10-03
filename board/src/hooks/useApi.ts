// Data from the brain API, outside or around a live meeting.
import type {
  Decision,
  Meeting,
  Person,
  Report,
  ReportProgress,
  TaskDraft,
  TeamSettings,
  TranscriptSegment,
  Voice,
} from "@moe/contracts";
import { notImplemented } from "@/lib/stub";

export const useMeetings = (): Meeting[] => notImplemented("useMeetings");
export const useMeeting = (meetingId: string): Meeting | null => notImplemented("useMeeting");
export const useMembers = (): Person[] => notImplemented("useMembers");
export const useReport = (meetingId: string): Report | null => notImplemented("useReport");
export const useReportProgress = (meetingId: string): ReportProgress | null => notImplemented("useReportProgress");
export const useSavedTranscript = (meetingId: string): TranscriptSegment[] => notImplemented("useSavedTranscript");
export const useDecisions = (query?: string): Decision[] => notImplemented("useDecisions");
export const useTasks = (ownerId?: string): TaskDraft[] => notImplemented("useTasks");
export const useSettings = (): TeamSettings | null => notImplemented("useSettings");
export const useVoices = (): Voice[] => notImplemented("useVoices");
