// Brain HTTP client (NEXT_PUBLIC_API_URL). One function per brain route; every call sends the
// session from lib/auth.ts. A 401 signs out (the app then goes to the sign-in page).
import type {
  Agenda,
  AgendaRewriteRequest,
  AgendaRewriteResponse,
  AgendaSuggestions,
  AgendaUpdate,
  Answer,
  AskRequest,
  ConnectorStatus,
  CreateAccountRequest,
  CreateAccountResponse,
  ConnectorsUpdate,
  GitHubAccountConnect,
  JiraAccountConnect,
  CreateMeetingRequest,
  Decision,
  InviteRequest,
  JoinMeetingResponse,
  Meeting,
  MeetingPresence,
  PasswordChange,
  Person,
  ProfileUpdate,
  Report,
  ReportProgress,
  TaskDraft,
  TaskPushRequest,
  TaskPushResult,
  Team,
  TeamSettings,
  TranscriptSegment,
  Voice,
} from "@moe/contracts";
import { apiBase, apiUrl } from "./apiUrl";
import { authHeaders, signOut } from "./auth";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    detail: string,
    /** Seconds to wait, from Retry-After on a 429. */
    readonly retryAfter: number | null = null,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

/** "a minute", "3 minutes", "40 seconds". */
export function waitText(seconds: number): string {
  if (seconds < 60) return `${seconds} second${seconds === 1 ? "" : "s"}`;
  const minutes = Math.ceil(seconds / 60);
  return minutes === 1 ? "a minute" : `${minutes} minutes`;
}

/** One sentence a person can act on. Says what is missing instead of hiding it. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "Your session has expired. Sign in again.";
    if (error.status === 403) return error.message || "You're not a member of this team.";
    if (error.status === 404) return "Not found.";
    if (error.status === 429) return error.retryAfter ? `Too many attempts. Try again in ${waitText(error.retryAfter)}.` : error.message;
    if (error.status === 501) return "This isn't available yet: the server hasn't implemented it.";
    if (error.status === 503) return `This isn't available right now. ${error.message}`;
    return error.message;
  }
  if (error instanceof TypeError) return `Can't reach the server${apiBase() ? ` at ${apiBase()}` : ""}.`;
  return error instanceof Error ? error.message : "Something went wrong.";
}

/** True for a 401 about the session itself (the brain sends WWW-Authenticate: Bearer with it),
 * as opposed to a wrong current password on POST /auth/password. */
const sessionRejected = (res: Response) => res.status === 401 && (res.headers.get("WWW-Authenticate") ?? "").startsWith("Bearer");

async function send(method: string, path: string, body?: unknown, accept401 = false): Promise<Response> {
  const headers: Record<string, string> = { ...authHeaders() };
  let payload: BodyInit | undefined;
  if (body instanceof FormData) {
    payload = body; // the browser sets the multipart boundary
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(apiUrl(path), { method, headers, body: payload });
  if (res.ok) return res;
  if (res.status === 401 && (!accept401 || sessionRejected(res))) signOut();
  let detail = res.statusText || `Request failed (${res.status})`;
  try {
    const json = await res.json();
    if (typeof json?.detail === "string") detail = json.detail;
  } catch {
    // not JSON; keep the status text
  }
  throw new ApiError(res.status, detail, Number(res.headers.get("Retry-After")) || null);
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await send(method, path, body);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const get = <T>(path: string) => request<T>("GET", path);
const id = encodeURIComponent;
const query = (params: Record<string, string | boolean | undefined>) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) if (value !== undefined && value !== "") search.set(key, String(value));
  const text = search.toString();
  return text ? `?${text}` : "";
};

// auth (sign-in itself is lib/auth.ts)
/** 401 here means the current password is wrong and does not sign out; 204 on success. */
export const changePassword = async (body: PasswordChange): Promise<void> => {
  await send("POST", "/auth/password", body, true);
};

// meetings
export const createMeeting = (body: CreateMeetingRequest) => request<Meeting>("POST", "/meetings", body);
export const listMeetings = () => get<Meeting[]>("/meetings");
export const getMeeting = (meetingId: string) => get<Meeting>(`/meetings/${id(meetingId)}`);
export const joinMeeting = (code: string) => request<JoinMeetingResponse>("POST", `/meetings/join/${id(code)}`);
/** Who is in the room now, for the lobby; participant_ids is everyone who ever joined. */
export const getPresence = (meetingId: string) => get<MeetingPresence>(`/meetings/${id(meetingId)}/presence`);
export const endMeeting = (meetingId: string) => request<Meeting>("POST", `/meetings/${id(meetingId)}/end`);
export const invite = (meetingId: string, body: InviteRequest) =>
  request<Meeting>("POST", `/meetings/${id(meetingId)}/invitees`, body);
export const uninvite = (meetingId: string, personId: string) =>
  request<Meeting>("DELETE", `/meetings/${id(meetingId)}/invitees/${id(personId)}`);
/** Host only, before anyone joins: live translation of non-English speech (#106). */
export const setTranslation = (meetingId: string, translate: boolean) =>
  request<Meeting>("PUT", `/meetings/${id(meetingId)}/translation`, { translate });
export const askInMeeting = (meetingId: string, body: AskRequest) =>
  request<Answer>("POST", `/meetings/${id(meetingId)}/ask`, body);

// agenda
export const getAgenda = (meetingId: string) => get<Agenda>(`/meetings/${id(meetingId)}/agenda`);
export const updateAgenda = (meetingId: string, body: AgendaUpdate) =>
  request<Agenda>("PUT", `/meetings/${id(meetingId)}/agenda`, body);
export const rewriteAgendaItem = (body: AgendaRewriteRequest) => request<AgendaRewriteResponse>("POST", "/agenda/rewrite", body);
export const suggestAgenda = (meetingId: string) =>
  request<AgendaSuggestions>("POST", `/meetings/${id(meetingId)}/agenda/suggest`);

// history
export const getTranscript = (meetingId: string) => get<TranscriptSegment[]>(`/meetings/${id(meetingId)}/transcript`);
export const getReport = (meetingId: string) => get<Report>(`/meetings/${id(meetingId)}/report`);
export const getReportProgress = (meetingId: string) =>
  get<ReportProgress>(`/meetings/${id(meetingId)}/report/progress`);
/** Host only, when the write-up stopped or stalled; starts it over. */
export const retryReport = (meetingId: string) =>
  request<ReportProgress>("POST", `/meetings/${id(meetingId)}/report/retry`);
/** The summary read aloud as MP3; 503 when ElevenLabs is not configured. */
export const getReportAudio = async (meetingId: string): Promise<Blob> =>
  (await send("GET", `/meetings/${id(meetingId)}/report/audio`)).blob();
export const updateTask = (meetingId: string, task: TaskDraft) =>
  request<TaskDraft>("PATCH", `/meetings/${id(meetingId)}/tasks/${id(task.id)}`, task);
export const pushTasks = (meetingId: string, body: TaskPushRequest) =>
  request<TaskPushResult[]>("POST", `/meetings/${id(meetingId)}/tasks/push`, body);
export const listDecisions = (q?: string) => get<Decision[]>(`/decisions${query({ q })}`);
export const listTasks = (ownerId?: string, open?: boolean) =>
  get<TaskDraft[]>(`/tasks${query({ owner_id: ownerId, open: open || undefined })}`);
export const askHistory = (body: AskRequest) => request<Answer>("POST", "/ask", body);

// profile
export const getMe = () => get<Person>("/me");
export const updateMe = (body: ProfileUpdate) => request<Person>("PATCH", "/me", body);
/** JPG or PNG, at most 2 MB. */
export const uploadPhoto = (photo: Blob, filename = "photo.jpg") => {
  const form = new FormData();
  form.append("file", photo, filename);
  return request<Person>("POST", "/me/photo", form);
};
export const deletePhoto = () => request<Person>("DELETE", "/me/photo");
/** A teammate's photo; `photoUrl` is Person.photo_url, a path on the brain. */
export const getPhoto = async (photoUrl: string): Promise<Blob> => (await send("GET", photoUrl)).blob();

// team
export const getTeam = () => get<Team>("/team");
export const listMembers = () => get<Person[]>("/team/members");
/** Admin only. The generated password is in this response and nowhere else. */
export const createAccount = (body: CreateAccountRequest) => request<CreateAccountResponse>("POST", "/team/accounts", body);
export const getSettings = () => get<TeamSettings>("/settings");
export const updateSettings = (body: TeamSettings) => request<TeamSettings>("PUT", "/settings", body);
export const listConnectors = () => get<ConnectorStatus[]>("/settings/connectors");
/** Admins only: the whole connector choice, replacing the saved one. */
export const updateConnectors = (body: ConnectorsUpdate) => request<TeamSettings>("PUT", "/settings/connectors", body);
/** Admins only: connects the team's Jira account. The brain checks it against Jira first. */
export const connectJiraAccount = (body: JiraAccountConnect) => request<TeamSettings>("PUT", "/settings/jira/account", body);
/** Admins only: forgets the team's Jira account and its token. */
export const disconnectJiraAccount = () => request<TeamSettings>("DELETE", "/settings/jira/account");
/** Admins only: connects GitHub with a fine-grained token. The brain checks it with GitHub first. */
export const connectGitHubAccount = (body: GitHubAccountConnect) => request<TeamSettings>("PUT", "/settings/github/account", body);
/** Admins only: forgets the team's GitHub token; its repositories are then read without it. */
export const disconnectGitHubAccount = () => request<TeamSettings>("DELETE", "/settings/github/account");
export const listVoices = () => get<Voice[]>("/voices");
