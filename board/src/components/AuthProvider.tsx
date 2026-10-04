"use client";

import type { Person } from "@moe/contracts";
import { usePathname, useRouter } from "next/navigation";
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";
import { useMe, useMembers } from "@/hooks/useApi";
import { unknownPerson } from "@/lib/format";
import { onSessionChange, sessionToken, signOut } from "@/lib/auth";

type AuthStatus = "loading" | "signed_out" | "signed_in";

const AuthContext = createContext<AuthStatus>("loading");

/** Whether there is a session from the brain (lib/auth.ts), kept current across tabs. */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  useEffect(() => {
    const check = () => setStatus(sessionToken() ? "signed_in" : "signed_out");
    check();
    // A session expires on its own: look again now and then, not only when something changes.
    const timer = setInterval(check, 60_000);
    const stop = onSessionChange(check);
    return () => {
      clearInterval(timer);
      stop();
    };
  }, []);
  return <AuthContext.Provider value={status}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);

interface TeamValue {
  me: Person;
  email: string | null;
  members: Person[];
  /** Why the members list is missing, if it is. */
  membersError: Error | null;
  /** Any account by id; a labelled stand-in when the id is not a known member. */
  person: (id: string) => Person;
  reloadMe: () => void;
  /** Replace the signed-in user's own record after they changed it. */
  setMe: (person: Person) => void;
}

const TeamContext = createContext<TeamValue | null>(null);

export function useTeam(): TeamValue {
  const value = useContext(TeamContext);
  if (!value) throw new Error("useTeam needs a signed-in user (wrap the page in AuthGate).");
  return value;
}

function TeamProvider({ children }: { children: ReactNode }) {
  const meQuery = useMe();
  const membersQuery = useMembers();
  const [changed, setChanged] = useState<Person | null>(null);
  const { reload: reloadMeQuery } = meQuery;
  const reloadMe = useCallback(() => {
    setChanged(null);
    reloadMeQuery();
  }, [reloadMeQuery]);
  const loaded = changed ?? meQuery.data;
  const value = useMemo<TeamValue | null>(() => {
    if (!loaded) return null;
    // The members list may predate a change to your own name or photo.
    const members = (membersQuery.data ?? []).map((p) => (p.id === loaded.id ? loaded : p));
    return {
      me: loaded,
      email: loaded.email ?? null,
      members,
      membersError: membersQuery.error,
      person: (id) => (id === loaded.id ? loaded : (members.find((p) => p.id === id) ?? unknownPerson(id))),
      reloadMe,
      setMe: setChanged,
    };
  }, [loaded, membersQuery.data, membersQuery.error, reloadMe]);

  if (!value) {
    return (
      <main className="auth">
        <div className="auth-card">
          {meQuery.error ? (
            <>
              <Notice error={meQuery.error} onRetry={meQuery.reload}>
                Your account can&apos;t be loaded.
              </Notice>
              <button type="button" className="btn btn-outline" onClick={signOut}>
                <Icon name="log-out" />
                Sign out
              </button>
            </>
          ) : (
            <p className="muted-p">Loading…</p>
          )}
        </div>
      </main>
    );
  }
  return <TeamContext.Provider value={value}>{children}</TeamContext.Provider>;
}

/** Signed-in pages only. Everyone else goes to sign-in and comes back afterwards. */
export function AuthGate({ children }: { children: ReactNode }) {
  const status = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  useEffect(() => {
    if (status === "signed_out") router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [status, pathname, router]);
  if (status !== "signed_in") return null;
  return <TeamProvider>{children}</TeamProvider>;
}
