"use client";

import { useEffect, useRef, useState } from "react";
import { Icon, Spinner } from "@/components/ui/Icon";
import { ApiError, describeError, getReportAudio } from "@/lib/api";

export interface ListenButtonProps {
  meetingId: string;
}

/**
 * The speaker button beside the Summary heading: plays the report's summary, read word for word
 * in the agent's voice. Fetches the audio
 * with getReportAudio (the session header), plays it from a blob URL and revokes the URL when
 * done. Shows an unavailable state when the brain answers 503 (ElevenLabs not configured).
 * Report page only; never plays into a meeting.
 */
export function ListenButton({ meetingId }: ListenButtonProps) {
  const [state, setState] = useState<"idle" | "loading" | "playing">("idle");
  const [problem, setProblem] = useState("");
  const audio = useRef<HTMLAudioElement | null>(null);
  const url = useRef<string | null>(null);

  const release = () => {
    audio.current?.pause();
    audio.current = null;
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = null;
  };
  useEffect(() => release, []);

  const listen = async () => {
    if (state === "playing") {
      release();
      setState("idle");
      return;
    }
    setState("loading");
    setProblem("");
    try {
      const blob = await getReportAudio(meetingId);
      release();
      url.current = URL.createObjectURL(blob);
      const player = new Audio(url.current);
      audio.current = player;
      player.onended = () => {
        release();
        setState("idle");
      };
      await player.play();
      setState("playing");
    } catch (err) {
      release();
      setState("idle");
      setProblem(
        err instanceof ApiError && err.status === 503 ? "Listening isn't available: no voice is set up." : describeError(err),
      );
    }
  };

  const label = state === "playing" ? "Stop reading the summary" : state === "loading" ? "Getting the audio…" : "Listen to the summary";
  return (
    <span className="listen">
      <button
        type="button"
        className="icon-btn sm"
        onClick={() => void listen()}
        disabled={state === "loading"}
        aria-label={label}
        title={label}
      >
        {state === "loading" ? <Spinner /> : <Icon name={state === "playing" ? "x" : "volume-2"} />}
      </button>
      {problem && (
        <span role="status" className="note">
          {problem}
        </span>
      )}
    </span>
  );
}
