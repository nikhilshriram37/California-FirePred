-- Clear the Security Advisor error "RLS Disabled in Public" on spatial_ref_sys.
--
-- spatial_ref_sys is PostGIS's reference table of coordinate-system (EPSG)
-- definitions. The extension creates it in the public schema with RLS off, so
-- Supabase's linter flags it as "publicly accessible" even though it holds no
-- user data. Enabling RLS with no policy denies access to the API roles
-- (anon / authenticated) while leaving server-side and PostGIS usage unaffected
-- (the pipeline connects as service_role, and PostGIS internals run as the table
-- owner — both bypass RLS). Our own tables already have RLS (migrations 0001, 0003).
--
-- Run this in the Supabase SQL editor (Dashboard -> SQL Editor), as with 0001-0003.
-- If it errors with "must be owner of table spatial_ref_sys", run it from the SQL
-- editor's default role (which owns it); the editor has the required privilege.

alter table public.spatial_ref_sys enable row level security;

-- Note: we intentionally add NO policy. spatial_ref_sys is not read through the
-- public API by this project, so deny-all to anon/authenticated is the secure and
-- correct outcome. If a future client ever needs to read it over the API, add:
--   create policy "public read spatial_ref_sys"
--     on public.spatial_ref_sys for select using (true);
