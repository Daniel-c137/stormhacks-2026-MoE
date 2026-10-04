import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

let client: SupabaseClient | null = null;

/** False when the browser config is missing; sign-in is then unavailable, never simulated. */
export const supabaseConfigured = (): boolean => Boolean(url && anonKey);

/** Browser client for email/password auth (NEXT_PUBLIC_SUPABASE_*). */
export function supabaseBrowser(): SupabaseClient {
  if (!url || !anonKey) {
    throw new Error("Sign-in is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.");
  }
  client ??= createClient(url, anonKey);
  return client;
}
