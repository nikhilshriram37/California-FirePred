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

## Retraining loop
Two scheduled jobs build the growing, labeled dataset:
- **`score_daily.yml`** (13:00 UTC) — writes each day's features + prediction to
  `feature_history` / `risk_scores`.
- **`backfill_labels.yml`** (15:00 UTC) — re-labels the trailing 7 days by fusing
  three fire sources: **IRWIN/WFIGS** + **CAL FIRE** (confirmed incidents, primary)
  and **NASA FIRMS** (satellite heat, supplementary recall). Sets
  `feature_history.has_fire` (prediction-vs-reality outcome) and `label_source`
  (which source(s) confirmed each cell), and archives FIRMS detections to
  `active_fires`. The 7-day window catches late-arriving incidents/detections.
  Run manually: `python -m src.pipeline.backfill_labels --days 7`.

Then, weekly/biweekly: pull the labeled `feature_history`, concat with the historical
parquet, re-run `src.models.train`, and upload new artifacts. Spatial CV + MERRA-2
PM2.5 fold in here.

**Label fidelity:** the model trained on FPA-FOD official ignitions. The loop now
labels primarily from confirmed incident records — **IRWIN/WFIGS** (interagency,
FPA-FOD-aligned: discovery date, location, cause, size) and **CAL FIRE** — which
match the training definition far better than heat detection. **FIRMS** is kept as a
supplementary detector for small/unreported fires. `label_source` lets retraining
prefer confirmed labels or weight them above FIRMS-only ones. For periodic
high-fidelity benchmarking, use FPA-FOD updates + MTBS perimeters.
