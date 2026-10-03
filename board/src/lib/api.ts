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
  TaskDraft,
  TaskPushRequest,
  TaskPushResult,
  Team,
  TeamSettings,
  TranscriptSegment,
  Voice,
} from "@moe/contracts";
import { notImplemented } from "./stub";

// meetings
export const createMeeting = async (body: CreateMeetingRequest): Promise<Meeting> => notImplemented("createMeeting");
export const listMeetings = async (): Promise<Meeting[]> => notImplemented("listMeetings");
export const getMeeting = async (meetingId: string): Promise<Meeting> => notImplemented("getMeeting");
export const joinMeeting = async (code: string): Promise<JoinMeetingResponse> => notImplemented("joinMeeting");
export const endMeeting = async (meetingId: string): Promise<Meeting> => notImplemented("endMeeting");
export const askInMeeting = async (meetingId: string, body: AskRequest): Promise<Answer> => notImplemented("askInMeeting");
export const getAgenda = async (meetingId: string): Promise<Agenda> => notImplemented("getAgenda");
export const generateAgenda = async (meetingId: string): Promise<Agenda> => notImplemented("generateAgenda");

// history
export const getTranscript = async (meetingId: string): Promise<TranscriptSegment[]> => notImplemented("getTranscript");
export const getReport = async (meetingId: string): Promise<Report> => notImplemented("getReport");
export const getReportProgress = async (meetingId: string): Promise<ReportProgress> => notImplemented("getReportProgress");
export const updateTask = async (meetingId: string, task: TaskDraft): Promise<TaskDraft> => notImplemented("updateTask");
export const pushTasks = async (meetingId: string, body: TaskPushRequest): Promise<TaskPushResult[]> => notImplemented("pushTasks");
export const listDecisions = async (q?: string): Promise<Decision[]> => notImplemented("listDecisions");
export const listTasks = async (ownerId?: string): Promise<TaskDraft[]> => notImplemented("listTasks");
export const askHistory = async (body: AskRequest): Promise<Answer> => notImplemented("askHistory");

// team
export const getTeam = async (): Promise<Team> => notImplemented("getTeam");
export const listMembers = async (): Promise<Person[]> => notImplemented("listMembers");
export const getSettings = async (): Promise<TeamSettings> => notImplemented("getSettings");
export const updateSettings = async (body: TeamSettings): Promise<TeamSettings> => notImplemented("updateSettings");
export const listVoices = async (): Promise<Voice[]> => notImplemented("listVoices");
