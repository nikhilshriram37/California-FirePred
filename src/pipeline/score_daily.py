"""Daily live scoring: fetch live feeds -> features -> predict -> write snapshot.

This is the live counterpart to export_snapshot's historical replay. It produces
the same dashboard snapshot, but from *today's* gridMET + GOES-GLM lightning
(plus seasonal dryness), and persists every scored day to Supabase.

As well as the latest gridMET day, the run recovers any of the last few gridMET
days that were never scored. Without that, a missed or failed run lost those
weather days *permanently*: the pipeline only ever looked at gridMET's newest day,
so 2026-07-24 and 07-25 (155 fire cells between them) slipped through when the
publishing date jumped 07-23 -> 07-26, and nothing in the live path would ever go
back for them. Recovered days feed the retrain and the live scorecards, so the
catch-up window stays comfortably inside the label job's re-label window — a day
recovered after labelling has moved on is a day with no ground truth.

Run:  python -m src.pipeline.score_daily                 # full GLM scan (~7-8 min)
      python -m src.pipeline.score_daily --glm-sample 4  # faster, approx lightning
      python -m src.pipeline.score_daily --catchup-days 0  # latest day only
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd

from src.data_acquisition.config import PROCESSED_DIR, REFERENCE_DIR
from src.data_acquisition.fetch_glm import fetch_glm_lightning
from src.data_acquisition.fetch_live import dryness_for_month, fetch_gridmet_recent
from src.models.predict import load_model
from src.pipeline.build_live_features import build_live_features
from src.pipeline.snapshot import build_meta, day_to_feature_collection, write_snapshot
from src.pipeline.supabase_io import get_client, persist_daily

logger = logging.getLogger(__name__)

# How far back to look for unscored gridMET days. Must stay below
# backfill_labels.LABEL_WINDOW_DAYS so a recovered day is still labellable when it
# lands. Four days covers two full days of failed runs with room to spare; going
# wider mostly adds GLM scans for days the label job can no longer reach.
CATCHUP_DAYS = 4


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


def _score_target(grid: pd.DataFrame, gridmet: pd.DataFrame, target: pd.Timestamp,
                  glm_sample: int, *, mode: str, write: bool, persist: bool) -> dict:
    """Score one gridMET day: features -> prediction -> snapshot / Supabase rows."""
    # Seasonal dryness (TerraClimate has no live feed) + lightning for that day.
    # GLM is asked for an explicit end-instant rather than "the last 24 hours", so a
    # recovered day reads the archive window it actually belongs to.
    end_utc = dt.datetime(target.year, target.month, target.day, 23, 59, 59, tzinfo=dt.UTC)
    lightning = fetch_glm_lightning(grid, hours=24, end=end_utc, sample_every=glm_sample)

    day, target = build_live_features(grid, gridmet, dryness_for_month(int(target.month)),
                                      lightning, target_date=target)
    day = day.join(load_model().predict(day))

    meta = build_meta(day, target.strftime("%Y-%m-%d"), mode=mode,
                      source=f"{mode}: gridMET + GOES-GLM",
                      lightning_cells=int((day["lightning_count"] > 0).sum()))

    if write:
        write_snapshot(day_to_feature_collection(day), meta)
        logger.info("Wrote %s snapshot for %s -> %s", mode.upper(), meta["data_date"],
                    meta["tier_counts"])
    if persist:
        persist_daily(day, meta)  # no-op if Supabase isn't configured
    return meta


def unscored_recent(dates: list[pd.Timestamp], target: pd.Timestamp,
                    catchup_days: int) -> list[pd.Timestamp]:
    """gridMET days within the catch-up window that have no risk_scores rows, oldest first.

    Returns nothing when Supabase is unconfigured: without the record of what was
    already scored there is no way to tell a missed day from a scored one, and
    re-scoring the window blindly would rewrite good live predictions.
    """
    if catchup_days <= 0:
        return []
    client = get_client()
    if client is None:
        return []
    window = [d for d in dates if d < target and (target - d).days <= catchup_days]
    missing = []
    for d in sorted(window):
        ds = d.strftime("%Y-%m-%d")
        if not client.table("risk_scores").select("grid_id").eq("date", ds).limit(1).execute().data:
            missing.append(d)
    return missing


def score_daily(glm_sample: int = 1, write: bool = True, persist: bool = True,
                catchup_days: int = CATCHUP_DAYS) -> dict:
    grid = canonical_grid()

    # Live weather backbone -> determines the target (latest available) day. The same
    # frame covers the catch-up days: its window already reaches back far enough for
    # their 14-day rolling features.
    gridmet = fetch_gridmet_recent(grid)
    gridmet["date"] = pd.to_datetime(gridmet["date"])
    dates = sorted(gridmet["date"].unique())
    target = dates[-1]

    # Recover missed days first so the last write is the live day — the dashboard
    # takes the newest data_date, and the local snapshot is a single file.
    for missed in unscored_recent(dates, target, catchup_days):
        try:
            m = _score_target(grid, gridmet, missed, glm_sample,
                              mode="catchup", write=False, persist=persist)
            logger.info("recovered unscored day %s -> %s cells %s",
                        m["data_date"], m["n_cells"], m["tier_counts"])
        except Exception as e:
            # Recovering an old day is a bonus; it must never cost us today's scoring.
            logger.warning("catch-up for %s failed (%s) — continuing to the live day",
                           pd.Timestamp(missed).date(), e)

    return _score_target(grid, gridmet, target, glm_sample,
                         mode="live", write=write, persist=persist)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm-sample", type=int, default=1,
                    help="process every Nth GLM granule (1 = full day, slower)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-persist", action="store_true", help="skip Supabase write")
    ap.add_argument("--catchup-days", type=int, default=CATCHUP_DAYS,
                    help="also score unscored gridMET days this far back (0 to disable)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    meta = score_daily(glm_sample=args.glm_sample, write=not args.no_write,
                       persist=not args.no_persist, catchup_days=args.catchup_days)
    print(meta)


if __name__ == "__main__":
    main()
