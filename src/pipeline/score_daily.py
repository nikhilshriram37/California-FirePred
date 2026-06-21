"""Daily live scoring: fetch live feeds -> features -> predict -> write snapshot.

This is the live counterpart to export_snapshot's historical replay. It produces
the same dashboard snapshot, but from *today's* gridMET + GOES-GLM lightning
(plus seasonal dryness). Persisting results to Supabase + growing the dataset is
wired in a later phase; for now it writes the local snapshot the dashboard reads.

Run:  python -m src.pipeline.score_daily                 # full GLM scan (~7-8 min)
      python -m src.pipeline.score_daily --glm-sample 4  # faster, approx lightning
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.data_acquisition.config import PROCESSED_DIR, REFERENCE_DIR
from src.data_acquisition.fetch_glm import fetch_glm_lightning
from src.data_acquisition.fetch_live import dryness_for_month, fetch_gridmet_recent
from src.models.predict import load_model
from src.pipeline.build_live_features import build_live_features
from src.pipeline.snapshot import build_meta, day_to_feature_collection, write_snapshot
from src.pipeline.supabase_io import persist_daily

logger = logging.getLogger(__name__)


def canonical_grid() -> pd.DataFrame:
    """The cells the model knows. Prefers the small committed reference file
    (works in the cloud); falls back to the full parquet locally."""
    ref = REFERENCE_DIR / "grid_cells.json"
    if ref.exists():
        return pd.read_json(ref)
    return (
        pd.read_parquet(PROCESSED_DIR / "california_dataset.parquet",
                        columns=["grid_id", "lat_center", "lon_center"])
        .drop_duplicates("grid_id")
        .reset_index(drop=True)
    )


def score_daily(glm_sample: int = 1, write: bool = True, persist: bool = True) -> dict:
    grid = canonical_grid()

    # 1. Live weather backbone -> determines the target (latest available) day.
    gridmet = fetch_gridmet_recent(grid)
    target = pd.to_datetime(gridmet["date"]).max()

    # 2. Seasonal dryness (TerraClimate has no live feed) + live lightning.
    dryness = dryness_for_month(int(target.month))
    lightning = fetch_glm_lightning(grid, hours=24, sample_every=glm_sample)

    # 3. Assemble features and score.
    day, target = build_live_features(grid, gridmet, dryness, lightning, target_date=target)
    day = day.join(load_model().predict(day))

    geojson = day_to_feature_collection(day)
    meta = build_meta(day, target.strftime("%Y-%m-%d"), mode="live",
                      source="live: gridMET + GOES-GLM",
                      lightning_cells=int((day["lightning_count"] > 0).sum()))

    if write:
        write_snapshot(geojson, meta)
        logger.info("Wrote LIVE snapshot for %s -> %s", meta["data_date"], meta["tier_counts"])
    if persist:
        persist_daily(day, meta)  # no-op if Supabase isn't configured
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm-sample", type=int, default=1,
                    help="process every Nth GLM granule (1 = full day, slower)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-persist", action="store_true", help="skip Supabase write")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    meta = score_daily(glm_sample=args.glm_sample, write=not args.no_write, persist=not args.no_persist)
    print(meta)


if __name__ == "__main__":
    main()
