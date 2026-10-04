"use client";

import type { Meeting } from "@moe/contracts";
import { useState } from "react";

import { TranslationSwitch } from "@/components/TranslationSwitch";
import { useTeam } from "@/components/AuthProvider";
import { describeError, setTranslation } from "@/lib/api";

/**
 * The meeting's translation switch in the lobby. Only the host can change it, and only until
 * someone joins: the agent reads it once, when it opens the meeting (#106).
 */
export function LobbyTranslation({ meeting }: { meeting: Meeting }) {
  const { me } = useTeam();
  const [on, setOn] = useState(meeting.translate);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const open = (meeting.status === "scheduled" || meeting.status === "live") && meeting.participant_ids.length === 0;
  const isHost = me?.id === meeting.host_id;
  const note = !open ? "It can't change once someone has joined." : !isHost ? "Only the host can change it." : undefined;

  async function change(next: boolean) {
    setBusy(true);
    setError("");
    try {
      setOn((await setTranslation(meeting.id, next)).translate);
    } catch (err) {
      setError(`Translation wasn't changed. ${describeError(err)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="lobby-translation" aria-label="Translation">
      <TranslationSwitch checked={on} onChange={isHost && open ? change : undefined} disabled={busy} note={note} />
      <span role="alert" className="err">
        {error}
      </span>
    </section>
  );
}
