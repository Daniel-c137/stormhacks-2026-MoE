// Brain HTTP client (NEXT_PUBLIC_API_URL). One function per brain route.
import type {
  Agenda,
  Answer,
  AskRequest,
  CreateMeetingRequest,
  Decision,
  JoinMeetingResponse,
  Meeting,
  Person,
  Report,
  ReportProgress,
  RewriteTopicRequest,
  RewriteTopicResponse,
  TaskDraft,
  TaskPushRequest,
  TaskPushResult,
  Team,
  TeamSettings,
  TranscriptSegment,
  UpdateAgendaRequest,
  UpdateProfileRequest,
  Voice,
} from "@moe/contracts";
import { supabaseBrowser, supabaseConfigured } from "./supabase";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

/** One sentence a person can act on. Says what is missing instead of hiding it. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 501) return "This isn't available yet: the server hasn't implemented it.";
    if (error.status === 401) return "Your session has expired. Sign in again.";
    if (error.status === 403) return "You're not a member of this meeting's team.";
    if (error.status === 404) return "Not found.";
    return error.message;
  }
  if (error instanceof TypeError) return `Can't reach the server${API_URL ? ` at ${API_URL}` : ""}.`;
  return error instanceof Error ? error.message : "Something went wrong.";
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  if (!API_URL) throw new Error("NEXT_PUBLIC_API_URL is not set.");
  const headers: Record<string, string> = {};
  if (supabaseConfigured()) {
    const { data } = await supabaseBrowser().auth.getSession();
    if (data.session) headers.Authorization = `Bearer ${data.session.access_token}`;
  }
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText || `Request failed (${res.status})`;
    try {
      const json = await res.json();
      if (typeof json?.detail === "string") detail = json.detail;
    } catch {
      // not JSON; keep the status text
    }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const get = <T>(path: string) => request<T>("GET", path);
const id = encodeURIComponent;

// meetings
export const createMeeting = (body: CreateMeetingRequest) => request<Meeting>("POST", "/meetings", body);
export const listMeetings = () => get<Meeting[]>("/meetings");
export const getMeeting = (meetingId: string) => get<Meeting>(`/meetings/${id(meetingId)}`);
export const joinMeeting = (code: string) => request<JoinMeetingResponse>("POST", `/meetings/join/${id(code)}`);
export const endMeeting = (meetingId: string) => request<Meeting>("POST", `/meetings/${id(meetingId)}/end`);
export const askInMeeting = (meetingId: string, body: AskRequest) =>
  request<Answer>("POST", `/meetings/${id(meetingId)}/ask`, body);
export const getAgenda = (meetingId: string) => get<Agenda>(`/meetings/${id(meetingId)}/agenda`);
export const generateAgenda = (meetingId: string) => request<Agenda>("POST", `/meetings/${id(meetingId)}/agenda`);
export const updateAgenda = (meetingId: string, body: UpdateAgendaRequest) =>
  request<Agenda>("PUT", `/meetings/${id(meetingId)}/agenda`, body);
export const rewriteTopic = (meetingId: string, body: RewriteTopicRequest) =>
  request<RewriteTopicResponse>("POST", `/meetings/${id(meetingId)}/agenda/rewrite`, body);

// history
export const getTranscript = (meetingId: string) => get<TranscriptSegment[]>(`/meetings/${id(meetingId)}/transcript`);
export const getReport = (meetingId: string) => get<Report>(`/meetings/${id(meetingId)}/report`);
export const getReportProgress = (meetingId: string) =>
  get<ReportProgress>(`/meetings/${id(meetingId)}/report/progress`);
export const updateTask = (meetingId: string, task: TaskDraft) =>
  request<TaskDraft>("PATCH", `/meetings/${id(meetingId)}/tasks/${id(task.id)}`, task);
export const pushTasks = (meetingId: string, body: TaskPushRequest) =>
  request<TaskPushResult[]>("POST", `/meetings/${id(meetingId)}/tasks/push`, body);
export const listDecisions = (q?: string) => get<Decision[]>(`/decisions${q ? `?q=${id(q)}` : ""}`);
export const listTasks = (ownerId?: string) => get<TaskDraft[]>(`/tasks${ownerId ? `?owner_id=${id(ownerId)}` : ""}`);
export const askHistory = (body: AskRequest) => request<Answer>("POST", "/ask", body);

// team
export const getMe = () => get<Person>("/me");
export const updateMe = (body: UpdateProfileRequest) => request<Person>("PATCH", "/me", body);
export const getTeam = () => get<Team>("/team");
export const listMembers = () => get<Person[]>("/team/members");
export const getSettings = () => get<TeamSettings>("/settings");
export const updateSettings = (body: TeamSettings) => request<TeamSettings>("PUT", "/settings", body);
export const listVoices = () => get<Voice[]>("/voices");
