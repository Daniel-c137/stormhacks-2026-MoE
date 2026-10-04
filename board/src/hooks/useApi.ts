"use client";

// Data from the brain API, outside or around a live meeting.
import type {
  Agenda,
  ConnectorStatus,
  Decision,
  Meeting,
  MeetingPresence,
  Person,
  Report,
  ReportProgress,
  TaskDraft,
  TeamSettings,
  TranscriptSegment,
  Voice,
} from "@moe/contracts";
import { useEffect, useReducer, useRef, useState } from "react";
import * as api from "@/lib/api";

export interface Query<T> {
  data: T | undefined;
  /** Set when the last request failed. Show it; never substitute made-up data. */
  error: Error | null;
  loading: boolean;
  reload: () => void;
}

/** Fetch on mount and whenever `key` changes; `key === null` skips; `refreshMs` polls. */
export function useQuery<T>(key: string | null, fetcher: () => Promise<T>, refreshMs = 0): Query<T> {
  const [state, setState] = useState<{ key: string | null; data?: T; error: Error | null; loading: boolean }>({
    key,
    error: null,
    loading: key !== null,
  });
  const [nonce, reload] = useReducer((n: number) => n + 1, 0);
  const latest = useRef(fetcher);
  useEffect(() => {
    latest.current = fetcher;
  });
  useEffect(() => {
    if (key === null) return;
    let cancelled = false;
    const run = () =>
      latest.current().then(
        (data) => {
          if (!cancelled) setState({ key, data, error: null, loading: false });
        },
        (error: Error) => {
          // Keep the last good data for this key while a refresh fails.
          if (!cancelled) setState((s) => ({ key, data: s.key === key ? s.data : undefined, error, loading: false }));
        },
      );
    void run();
    const timer = refreshMs ? setInterval(run, refreshMs) : undefined;
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [key, nonce, refreshMs]);
  const current = state.key === key;
  return {
    data: current ? state.data : undefined,
    error: current ? state.error : null,
    loading: key !== null && (!current || state.loading),
    reload,
  };
}

export const useMeetings = (refreshMs = 0): Query<Meeting[]> => useQuery("meetings", api.listMeetings, refreshMs);
export const useMeeting = (meetingId: string, refreshMs = 0): Query<Meeting> =>
  useQuery(`meeting:${meetingId}`, () => api.getMeeting(meetingId), refreshMs);
export const usePresence = (meetingId: string | null, refreshMs = 0): Query<MeetingPresence> =>
  useQuery(meetingId && `presence:${meetingId}`, () => api.getPresence(meetingId ?? ""), refreshMs);
export const useMe = (): Query<Person> => useQuery("me", api.getMe);
export const useMembers = (): Query<Person[]> => useQuery("members", api.listMembers);
export const useAgenda = (meetingId: string | null): Query<Agenda> =>
  useQuery(meetingId && `agenda:${meetingId}`, () => api.getAgenda(meetingId ?? ""));
export const useReport = (meetingId: string): Query<Report> =>
  useQuery(`report:${meetingId}`, () => api.getReport(meetingId));
export const useReportProgress = (meetingId: string, refreshMs = 0): Query<ReportProgress> =>
  useQuery(`progress:${meetingId}`, () => api.getReportProgress(meetingId), refreshMs);
export const useSavedTranscript = (meetingId: string): Query<TranscriptSegment[]> =>
  useQuery(`transcript:${meetingId}`, () => api.getTranscript(meetingId));
export const useDecisions = (query?: string): Query<Decision[]> =>
  useQuery(`decisions:${query ?? ""}`, () => api.listDecisions(query));
export const useTasks = (ownerId?: string): Query<TaskDraft[]> =>
  useQuery(`tasks:${ownerId ?? ""}`, () => api.listTasks(ownerId));
export const useSettings = (): Query<TeamSettings> => useQuery("settings", api.getSettings);
export const useVoices = (): Query<Voice[]> => useQuery("voices", api.listVoices);
export const useConnectors = (): Query<ConnectorStatus[]> => useQuery("connectors", api.listConnectors);
