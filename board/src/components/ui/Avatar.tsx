"use client";

import type { Person } from "@moe/contracts";
import type { CSSProperties } from "react";
import { usePhoto } from "@/hooks/usePhoto";
import { cx } from "@/lib/format";

// Colour by initials, so each person keeps one colour everywhere.
const COLORS = ["#4F7DFF", "#0FA295", "#D9468F", "#D98E1C", "#7C5CFA", "#23A05E", "#2F95D3", "#8A6247", "#5B66E0"];

export function avatarColor(initials: string): string {
  const code = initials.toUpperCase().charCodeAt(0);
  return Number.isNaN(code) || code < 65 ? "#7A8190" : COLORS[(code - 65) % COLORS.length];
}

export interface AvatarProps {
  person: Pick<Person, "name" | "initials" | "photo_url">;
  size?: "sm" | "lg";
  /** The signed-in user, drawn in the accent colour when they have no photo. */
  me?: boolean;
  title?: string;
}

export function Avatar({ person, size, me, title }: AvatarProps) {
  // The initials show until the photo has loaded, and if it can't be.
  const photo = usePhoto(person.photo_url);
  const style: CSSProperties = photo
    ? { backgroundImage: `url("${photo}")`, backgroundSize: "cover", backgroundPosition: "center" }
    : me
      ? {}
      : { background: avatarColor(person.initials) };
  return (
    <span
      className={cx("av", size, me && !photo && "av-me")}
      style={style}
      title={title}
      role={photo ? "img" : undefined}
      aria-label={photo ? person.name : undefined}
      aria-hidden={photo ? undefined : true}
    >
      {photo ? null : person.initials}
    </span>
  );
}
