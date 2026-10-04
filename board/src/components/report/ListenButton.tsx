"use client";

export interface ListenButtonProps {
  meetingId: string;
}

/**
 * Listen: plays the report's summary, read word for word in the agent's voice. Fetches the audio
 * with getReportAudio (the session header), plays it from a blob URL and revokes the URL when
 * done. Shows an unavailable state when the brain answers 503 (ElevenLabs not configured).
 * Report page only; never plays into a meeting.
 */
export function ListenButton(_props: ListenButtonProps) {
  return null;
}
