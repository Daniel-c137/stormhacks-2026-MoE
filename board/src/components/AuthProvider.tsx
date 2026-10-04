"use client";

import type { Person } from "@moe/contracts";
import type { Session } from "@supabase/supabase-js";
import { usePathname, useRouter } from "next/navigation";
import { createContext, type ReactNode, useContext, useEffect, useMemo, useState } from "react";
import { useMe, useMembers } from "@/hooks/useApi";
import { initialsOf, shortOf, unknownPerson } from "@/lib/format";
import { supabaseBrowser, supabaseConfigured } from "@/lib/supabase";

type AuthState =
  | { status: "loading" | "unconfigured" | "signed_out" }
  | { status: "signed_in"; session: Session };

const AuthContext = createContext<AuthState>({ status: "loading" });

/** Tracks the Supabase session for the whole app. */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ status: supabaseConfigured() ? "loading" : "unconfigured" });
  useEffect(() => {
    if (!supabaseConfigured()) return;
    const auth = supabaseBrowser().auth;
    const apply = (session: Session | null) =>
      setState(session ? { status: "signed_in", session } : { status: "signed_out" });
    void auth.getSession().then(({ data }) => apply(data.session));
    const { data } = auth.onAuthStateChange((_event, session) => apply(session));
    return () => data.subscription.unsubscribe();
  }, []);
  return <AuthContext.Provider value={state}>{children}</AuthContext.Provider>;
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
}

const TeamContext = createContext<TeamValue | null>(null);

export function useTeam(): TeamValue {
  const value = useContext(TeamContext);
  if (!value) throw new Error("useTeam needs a signed-in user (wrap the page in AuthGate).");
  return value;
}

/** The account as Supabase knows it, used until the brain returns the team's own record. */
function personFromSession(session: Session): Person {
  const meta = session.user.user_metadata as { name?: unknown; full_name?: unknown };
  const fromMeta = typeof meta.name === "string" ? meta.name : typeof meta.full_name === "string" ? meta.full_name : "";
  const name = fromMeta.trim() || session.user.email?.split("@")[0] || "You";
  return { id: session.user.id, name, short: shortOf(name), initials: initialsOf(name) };
}

function TeamProvider({ session, children }: { session: Session; children: ReactNode }) {
  const meQuery = useMe();
  const membersQuery = useMembers();
  const value = useMemo<TeamValue>(() => {
    const members = membersQuery.data ?? [];
    const me = meQuery.data ?? members.find((p) => p.id === session.user.id) ?? personFromSession(session);
    return {
      me,
      email: session.user.email ?? null,
      members,
      membersError: membersQuery.error,
      person: (id) => (id === me.id ? me : (members.find((p) => p.id === id) ?? unknownPerson(id))),
      reloadMe: meQuery.reload,
    };
  }, [meQuery.data, meQuery.reload, membersQuery.data, membersQuery.error, session]);
  return <TeamContext.Provider value={value}>{children}</TeamContext.Provider>;
}

/** Signed-in pages only. Everyone else goes to sign-in and comes back afterwards. */
export function AuthGate({ children }: { children: ReactNode }) {
  const auth = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const needsSignIn = auth.status === "signed_out" || auth.status === "unconfigured";
  useEffect(() => {
    if (needsSignIn) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [needsSignIn, pathname, router]);
  if (auth.status !== "signed_in") return null;
  return <TeamProvider session={auth.session}>{children}</TeamProvider>;
}
