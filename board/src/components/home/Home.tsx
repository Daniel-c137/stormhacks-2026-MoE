"use client";

import type { Meeting } from "@moe/contracts";
import { useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { Icon } from "@/components/ui/Icon";
import { useMeetings } from "@/hooks/useApi";
import { JoinMeetingModal, NewMeetingModal } from "./MeetingModals";
import { Rail } from "./Rail";

function greeting(now: Date): string {
  const h = now.getHours();
  return h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
}

/** New meeting, join by code or link, today's and earlier meetings, and questions across meeting history. */
export function Home() {
  const { me } = useTeam();
  const meetings = useMeetings(15_000);
  const [modal, setModal] = useState<"new" | "join" | null>(null);
  const [now] = useState(() => new Date());

  const onScheduled = (_meeting: Meeting) => {
    setModal(null);
    meetings.reload();
  };

  return (
    <div className="home">
      <main className="home-main">
        <section className="home-hero" aria-labelledby="h-greet">
          <p className="eyebrow">{now.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })}</p>
          <h1 id="h-greet" className="hero-title">
            {greeting(now)}, {me.short}.
          </h1>
          <div className="home-actions">
            <button type="button" className="btn btn-primary btn-lg" onClick={() => setModal("new")}>
              <Icon name="plus" />
              New meeting
            </button>
            <button type="button" className="btn btn-outline btn-lg" onClick={() => setModal("join")}>
              <Icon name="log-in" />
              Join meeting
            </button>
          </div>
        </section>
        <Rail meetings={meetings} now={now} />
      </main>
      {modal === "new" && <NewMeetingModal onClose={() => setModal(null)} onScheduled={onScheduled} />}
      {modal === "join" && <JoinMeetingModal meetings={meetings.data} onClose={() => setModal(null)} />}
    </div>
  );
}
