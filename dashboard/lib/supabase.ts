import { createClient, type SupabaseClient } from "@supabase/supabase-js";

/**
 * Server-side Supabase client. Returns null when no credentials are configured,
 * which is the signal for the data layer to fall back to the local snapshot.
 * Uses the service-role key (server only — never import this into a client
 * component).
 */
export function getServerSupabase(): SupabaseClient | null {
  const url = process.env.SUPABASE_URL ?? process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key =
    process.env.SUPABASE_SERVICE_ROLE_KEY ?? process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return null;
  return createClient(url, key, { auth: { persistSession: false } });
}
