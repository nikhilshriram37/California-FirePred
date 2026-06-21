# Supabase setup (data layer)

The dashboard reads risk + metadata from Supabase in production and falls back to
the local snapshot in dev. The daily scorer (`src.pipeline.score_daily`) writes to
it when credentials are present, and silently skips when they're not.

## One-time setup (when you create the project)

1. Create a free project at https://supabase.com → note the **Project URL**, the
   **anon** key, and the **service-role** key (Settings → API).
2. Apply the schema — either:
   - **SQL editor:** paste `migrations/0001_init.sql` and run, or
   - **CLI:** `supabase link --project-ref <ref>` then `supabase db push`
3. Set environment variables:

   **Pipeline / GitHub Actions** (service role — full write):
   ```
   SUPABASE_URL=https://<ref>.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=<service-role key>
   ```
   **Dashboard / Vercel** (server-side reads use the service role; the public
   anon vars are optional unless you query from the browser):
   ```
   SUPABASE_URL=https://<ref>.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=<service-role key>
   NEXT_PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key>
   ```
4. Install the Python client for persistence: `pip install -e ".[live]"`

## What gets written each run
- `grid_cells` — static cell geometry (idempotent)
- `risk_scores` — one row per cell per day (the choropleth)
- `feature_history` — the streamed features (grows the training dataset; `has_fire`
  is backfilled later once outcomes are known)
- `risk_meta` — snapshot metadata; the latest row is what the dashboard shows

## Note on free-tier size
The free Postgres tier is ~500 MB — it can't hold the full 4M-row history. Keep
operational/recent data here; the full training set stays as parquet, and
retraining concatenates parquet + the streamed `feature_history`.
