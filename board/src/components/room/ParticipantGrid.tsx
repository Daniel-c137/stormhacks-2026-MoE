"use client";

import type { Participant, Person } from "@moe/contracts";
import { type TrackReference, VideoTrack, useRoomContext } from "@livekit/components-react";
import { Track } from "livekit-client";
import type { CSSProperties } from "react";
import { Avatar } from "@/components/ui/Avatar";
import { Icon } from "@/components/ui/Icon";

export interface ParticipantGridProps {
  participants: Participant[];
  meId: string;
  /** Who fills the stage: the current speaker. A shared screen always takes it instead. */
  featuredId: string | null;
  /** Name, initials and photo for a participant. */
  personOf: (participant: Participant) => Person;
}

interface Tile {
  key: string;
  participant: Participant;
  screen: boolean;
  trackRef: TrackReference | null;
}

/** People only; the agent has no video tile. */
export function ParticipantGrid({ participants, meId, featuredId, personOf }: ParticipantGridProps) {
  const room = useRoomContext();
  const tiles: Tile[] = [];
  for (const participant of participants) {
    if (participant.is_agent) continue;
    const lk = participant.id === meId ? room.localParticipant : room.getParticipantByIdentity(participant.id);
    const refFor = (source: Track.Source): TrackReference | null => {
      const publication = lk?.getTrackPublication(source);
      return lk && publication?.track && !publication.isMuted ? { participant: lk, publication, source } : null;
    };
    const screen = refFor(Track.Source.ScreenShare);
    if (screen) tiles.push({ key: `${participant.id}:screen`, participant, screen: true, trackRef: screen });
    tiles.push({ key: participant.id, participant, screen: false, trackRef: refFor(Track.Source.Camera) });
  }

  const featured =
    tiles.length > 1 ? (tiles.find((t) => t.screen) ?? tiles.find((t) => t.participant.id === featuredId) ?? null) : null;
  const grid: Record<string, number> = featured
    ? { "--rows": Math.max(tiles.length - 1, 1) }
    : { "--cols": Math.ceil(Math.sqrt(tiles.length)) };
  const layout = grid as CSSProperties;

  return (
    <div className="pgrid" data-featured={Boolean(featured)} data-count={tiles.length} style={layout}>
      {tiles.map(({ key, participant: p, screen, trackRef }) => {
        const who = personOf(p);
        const you = p.id === meId;
        const name = `${who.name}${you ? " (you)" : ""}`;
        const speaking = p.is_speaking && p.mic_on && !screen;
        const label = screen
          ? `${name}'s screen`
          : [
              `${who.name}${p.role === "host" ? ", host" : ""}`,
              p.mic_on ? "mic on" : "muted",
              !p.cam_on && "camera off",
              speaking && "speaking",
              p.hand_raised && "hand raised",
            ]
              .filter(Boolean)
              .join(", ");
        return (
          <div
            key={key}
            className="ptile"
            role="group"
            aria-label={label}
            data-camoff={!trackRef}
            data-featured={featured?.key === key}
            data-speaking={speaking}
            data-screen={screen}
            data-mirror={you && !screen}
          >
            {trackRef ? <VideoTrack trackRef={trackRef} /> : <Avatar person={who} />}
            {speaking && <div className="speaking" aria-hidden="true" />}
            {p.hand_raised && !screen && (
              <span className="hand">
                <Icon name="hand" />
                <span>Hand raised</span>
              </span>
            )}
            <div className="name">
              <Icon name={screen ? "monitor-up" : p.mic_on ? "mic" : "mic-off"} />
              <span>{screen ? `${name}'s screen` : name}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
