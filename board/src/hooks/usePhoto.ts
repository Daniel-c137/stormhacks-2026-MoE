"use client";

import { useEffect, useState } from "react";
import { getPhoto } from "@/lib/api";

// Person.photo_url is a path on the brain that needs the session, so it can't go straight into an
// <img> or a CSS url(). Each photo is fetched once with the session and shown from a blob URL; the
// path changes whenever the photo does, so a cached one is never stale.
const cache = new Map<string, Promise<string | null>>();

function load(path: string): Promise<string | null> {
  let pending = cache.get(path);
  if (!pending) {
    pending = getPhoto(path).then(
      (blob) => URL.createObjectURL(blob),
      () => {
        cache.delete(path); // try again next time; meanwhile the initials show
        return null;
      },
    );
    cache.set(path, pending);
  }
  return pending;
}

/** A URL an <img> can show for `photoUrl`, or null while loading or when there is none. Local
 * previews (data: and blob: URLs) are used as they are. */
export function usePhoto(photoUrl: string | null | undefined): string | null {
  const local = photoUrl && /^(data|blob):/.test(photoUrl) ? photoUrl : null;
  const [loaded, setLoaded] = useState<{ path: string; url: string | null } | null>(null);
  useEffect(() => {
    if (!photoUrl || local) return;
    let cancelled = false;
    void load(photoUrl).then((url) => {
      if (!cancelled) setLoaded({ path: photoUrl, url });
    });
    return () => {
      cancelled = true;
    };
  }, [photoUrl, local]);
  if (local) return local;
  return photoUrl && loaded?.path === photoUrl ? loaded.url : null;
}
