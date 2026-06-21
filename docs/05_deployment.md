# Deployment guide

Architecture: **Vercel** hosts the dashboard, **Supabase** stores the data, and a
**GitHub Actions** cron runs the daily Python scorer that writes to Supabase.

```
GitHub Actions (daily)  ──score_daily──▶  Supabase (risk_scores, feature_history, risk_meta)
                                              ▲ reads
                                              │
Vercel (Next.js dashboard)  ──────────────────┘   + live FIRMS fires (direct)
```

## 1. Supabase (data)
1. Create a project at supabase.com.
2. SQL editor → paste `supabase/migrations/0001_init.sql` → Run.
3. Settings → API → copy **Project URL**, **anon key**, **service-role key**.

## 2. GitHub (code + cron)
1. Create an empty repo, then from this folder:
   ```bash
   git remote add origin git@github.com:<you>/<repo>.git
   git push -u origin main
   ```
2. Repo → Settings → Secrets and variables → Actions → add:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `NASA_FIRMS_MAP_KEY`
3. Actions tab → run **Daily wildfire scoring** once (`workflow_dispatch`) to
   populate Supabase. (It also runs daily at 13:00 UTC.)

## 3. Vercel (dashboard)
1. Import the GitHub repo.
2. **Root Directory → `dashboard`** (important — the app is in a subfolder).
3. Environment variables:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `NASA_FIRMS_MAP_KEY`
4. Deploy. The dashboard reads the latest `risk_meta` / `risk_scores` from Supabase;
   if those are empty it falls back to the committed snapshot.

## Local development
```bash
# live nowcast written locally + (optionally) to Supabase
python -m src.pipeline.score_daily
# dashboard
npm install --prefix dashboard && npm run dev --prefix dashboard
```
Without Supabase env vars the dashboard serves the committed snapshot and the
scorer skips persistence — everything still runs.

## Retraining loop (later)
`feature_history` accumulates daily. Periodically: pull it, backfill `has_fire`
from fire records, concat with the historical parquet, re-run `src.models.train`,
and upload the new artifacts. Spatial CV + MERRA-2 PM2.5 fold in here.
