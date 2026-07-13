"""Backfill missing historical days into Supabase (score, then label).

Reconstructs days the live cron missed — e.g. the gridMET-freeze gap 2026-06-20..
07-03 when the cron kept re-scoring a frozen date instead of advancing. Uses the
SAME current model + corrected topography + full 4169-cell grid as recent live
days, so the accumulated record is continuous AND on consistent feature footing.

ADDITIVE + IDEMPOTENT: skips any date that already has risk_scores, so existing
live predictions are never overwritten. gridMET is downloaded once (a wide window
covering every target's rolling lookback); GOES-GLM lightning is pulled per day
from the public archive (sampled for speed — a low-importance feature). After
scoring, each newly-created day is labeled from the fused fire sources.

Run: python -m scripts.backfill_history --start 2026-06-20 --end 2026-07-03
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd

from src.data_acquisition.config import REGIONS
from src.data_acquisition.fetch_glm import fetch_glm_lightning
from src.data_acquisition.fetch_live import GRIDMET_VARS, dryness_for_month
from src.models.predict import load_model
from src.preprocessing.build_dataset import fetch_gridmet_for_grid
from src.pipeline.backfill_labels import (_grid, _has_label_source, backfill_date,
                                          fetch_calfire_ca, fetch_irwin_ca)
from src.pipeline.build_live_features import build_live_features
from src.pipeline.score_daily import canonical_grid
from src.pipeline.snapshot import build_meta
from src.pipeline.supabase_io import get_client, persist_daily

logger = logging.getLogger(__name__)


def cached_gridmet(grid, end: dt.date, days: int) -> pd.DataFrame:
    """Recent gridMET window from the CACHED current-year file (no forced refresh).

    Backfill targets are historical dates already present in the cached file, so we
    read it (refresh_current=False) instead of re-downloading — the live path's
    refresh is only needed to advance the latest day, and it hammers a flaky host.
    """
    df = fetch_gridmet_for_grid(grid, end.year, variables=GRIDMET_VARS,
                                bbox=REGIONS["california"], refresh_current=False)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp(end)
    start = cutoff - pd.Timedelta(days=days + 5)
    win = df[(df["date"] >= start) & (df["date"] <= cutoff)].copy()
    logger.info("gridMET (cached): %s rows, %s..%s", f"{len(win):,}",
                win["date"].min().date(), win["date"].max().date())
    return win


def _already_scored(client, ds: str) -> bool:
    return bool(client.table("risk_scores").select("grid_id").eq("date", ds).limit(1).execute().data)


def score_day(grid, gridmet, D: dt.date, glm_sample: int) -> dict:
    end_utc = dt.datetime(D.year, D.month, D.day, 23, 59, 59, tzinfo=dt.UTC)
    lightning = fetch_glm_lightning(grid, hours=24, end=end_utc, sample_every=glm_sample)
    day, _ = build_live_features(grid, gridmet, dryness_for_month(D.month), lightning,
                                 target_date=pd.Timestamp(D))
    day = day.join(load_model().predict(day))
    meta = build_meta(day, D.strftime("%Y-%m-%d"), mode="backfill",
                      source="backfill: gridMET + GOES-GLM (archive)",
                      lightning_cells=int((day["lightning_count"] > 0).sum()))
    persist_daily(day, meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--glm-sample", type=int, default=8, help="process every Nth GLM granule")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    client = get_client()
    if client is None:
        raise SystemExit("Supabase not configured (.env.local)")

    grid = canonical_grid()
    # One cached gridMET read covering every target's 14-day rolling lookback.
    gridmet = cached_gridmet(grid, end, days=(end - start).days + 30)

    scored = []
    for D in dates:
        ds = D.strftime("%Y-%m-%d")
        if _already_scored(client, ds):
            logger.info("%s already scored — skipping (additive)", ds)
            continue
        meta = score_day(grid, gridmet, D, args.glm_sample)
        logger.info("scored %s -> %s cells %s", ds, meta["n_cells"], meta["tier_counts"])
        scored.append(D)

    if not scored:
        logger.info("nothing new to score"); return

    # Label only the days we just created (existing days keep their live labels).
    logger.info("labeling %d newly-scored days...", len(scored))
    irwin, calfire = fetch_irwin_ca(), fetch_calfire_ca()
    ws = _has_label_source(client)
    for D in scored:
        r = backfill_date(client, _grid(), D, irwin, calfire, ws)
        logger.info("labeled %s: %s", r["date"], r)
    logger.info("BACKFILL COMPLETE: %d days scored + labeled", len(scored))


if __name__ == "__main__":
    main()
