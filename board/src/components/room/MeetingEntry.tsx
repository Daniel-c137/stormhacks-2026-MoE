"use client";

import type { JoinMeetingResponse } from "@moe/contracts";
import { useState } from "react";
import { AuthGate } from "@/components/AuthProvider";
import { useMeetings } from "@/hooks/useApi";
import { ApiError, describeError, joinMeeting } from "@/lib/api";
import { type DeviceChoices, Lobby } from "./Lobby";
import { Room } from "./Room";

export interface MeetingEntryProps {
  code: string;
}

function Entry({ code }: MeetingEntryProps) {
  // The lobby needs the meeting's title before joining; the team's meetings are where a code resolves.
  const meetings = useMeetings();
  const meeting = meetings.data?.find((m) => m.code === code) ?? null;
  const [joined, setJoined] = useState<{ join: JoinMeetingResponse; choices: DeviceChoices } | null>(null);
  const [joining, setJoining] = useState(false);
  const [error, setError] = useState("");

  const onJoin = async (choices: DeviceChoices) => {
    setJoining(true);
    setError("");
    try {
      setJoined({ join: await joinMeeting(code), choices });
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 404
          ? "There's no meeting with that code."
          : err instanceof ApiError && err.status === 403
            ? "You can't join: you're not a member of this meeting's team."
            : `You couldn't join. ${describeError(err)}`,
      );
    } finally {
      setJoining(false);
    }
  };

  if (joined) return <Room join={joined.join} choices={joined.choices} />;
  return (
    <div className="app">
      <Lobby code={code} meeting={meeting} lookupError={meetings.error} joining={joining} error={error} onJoin={onJoin} />
    </div>
  );
}

/** Lobby until the user joins, then the room with the JoinMeetingResponse token. */
export function MeetingEntry({ code }: MeetingEntryProps) {
  return (
    <AuthGate>
      <Entry code={code} />
    </AuthGate>
  );
}
