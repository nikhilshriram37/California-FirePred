"""Build the three held-out evaluation panels the feature ablation grades on.

One fit configuration (train on 2018-19) is graded against three different regimes,
because the deployed model demonstrably behaves differently in each and an aggregate
over any one of them hides the others:

    holdout2020   2020 full year, 3,668-cell training grid, FPA-FOD ignitions.
                  The classic backtest — kept only as the era reference. It is the
                  most flattering and the least trustworthy of the three.
    live2026      2026-06-19..08-03 on the live 4,169-cell CA grid. Peak summer:
                  weather barely varies, fires cluster hard. This is the regime the
                  quiet-cell tier collapse was found in.
    autumn2025    2025-09-01..11-30, same grid. Shoulder season: 8x the weather
                  variance, fires more weather-driven and less spatially clustered.

Grading target is ``confirmed`` (agency IRWIN/CAL FIRE ignition records) for the two
recent panels, matching what the model is trained on. FIRMS satellite detections mark
where fire is *burning*, not where it *began*, and are ~48% of the fused positives.

Recency warm-up. The 2026 label record in Supabase starts 06-19, so a panel built from
it alone would begin with an artificially cold fire-recency prior and handicap the
recency block by construction. This pulls IRWIN year-to-date back to 2026-01-01 for
warm-up and switches to the panel's own confirmed labels inside the window, so the
in-window label definition is unchanged.

Run:  .venv/bin/python scripts/build_ablation_evalsets.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_acquisition.config import PROCESSED_DIR, PROJECT_ROOT, REFERENCE_DIR
from src.models.features import STATIC_FEATURES, TARGET_COL, merge_static_features
from src.models.recency import RECENCY_FEATURES, merge_recency
from src.preprocessing.build_dataset import engineer_features, fetch_gridmet_for_grid

logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
CELL_DEG = 0.1
HALF = 0.05

AUTUMN_START, AUTUMN_END = "2025-09-01", "2025-11-30"


def points_to_cell_days(pts: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Map (latitude, longitude, date) points to unique (grid_id, date) ignitions.

    Uses the same flooring rule as :func:`src.pipeline.backfill_labels.points_to_grid_ids`
    so an ignition lands in the cell the label pipeline would have put it in.
    """
    lookup = {(round(r.lat_center, 2), round(r.lon_center, 2)): int(r.grid_id)
              for r in grid.itertuples()}
    clat = np.round(np.floor(pts["latitude"].to_numpy() / CELL_DEG) * CELL_DEG + HALF, 2)
    clon = np.round(np.floor(pts["longitude"].to_numpy() / CELL_DEG) * CELL_DEG + HALF, 2)
    gid = [lookup.get((a, o)) for a, o in zip(clat, clon)]
    out = pd.DataFrame({"grid_id": gid, "date": pd.to_datetime(pts["date"].to_numpy())})
    out = out.dropna(subset=["grid_id"])
    out["grid_id"] = out["grid_id"].astype(int)
    return out.drop_duplicates().reset_index(drop=True)


def build_historical() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 2018-19 fitting panel and the 2020 holdout, prepared exactly as train.py does."""
    df = pd.read_parquet(PROCESSED_DIR / "california_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = merge_static_features(df)
    # Recency over the whole 2018-2020 panel so 2020 opens warm, exactly as training.
    df = merge_recency(df, df.loc[df[TARGET_COL] == 1, ["grid_id", "date"]])
    df["confirmed"] = df[TARGET_COL]          # FPA-FOD ignitions are agency records
    train = df[df["date"].dt.year.isin([2018, 2019])].copy()
    hold = df[df["date"].dt.year == 2020].copy()
    logger.info("train201819: %s rows, %s ignitions | holdout2020: %s rows, %s ignitions",
                f"{len(train):,}", int(train.confirmed.sum()),
                f"{len(hold):,}", int(hold.confirmed.sum()))
    return train, hold


def build_live2026(grid: pd.DataFrame) -> pd.DataFrame:
    """The corrected live panel plus a fully warmed fire-recency prior."""
    df = pd.read_parquet(EVAL_DIR / "live_features_corrected.parquet")
    df["date"] = pd.to_datetime(df["date"])

    # Warm-up from IRWIN YTD (pre-window only) + the panel's own confirmed labels
    # in-window, so the in-window label definition is untouched.
    ytd = pd.read_parquet(EVAL_DIR / "irwin_2026_ytd.parquet")
    warm = points_to_cell_days(ytd, grid)
    warm = warm[warm["date"] < df["date"].min()]
    inwin = df.loc[df["confirmed"] == 1, ["grid_id", "date"]]
    fires = pd.concat([warm, inwin], ignore_index=True).drop_duplicates()
    logger.info("live2026 recency history: %s pre-window + %s in-window ignitions",
                f"{len(warm):,}", f"{len(inwin):,}")

    # 150 days of warm-up covers 5 tau at tau=30 (a fire decays to <1% of its weight).
    df = merge_recency(df, fires, warmup_days=150)
    logger.info("live2026: %s rows, %s cells, %s confirmed ignitions",
                f"{len(df):,}", df.grid_id.nunique(), int(df.confirmed.sum()))
    return df


def build_autumn2025(grid: pd.DataFrame) -> pd.DataFrame:
    """Sep-Nov 2025 panel, rebuilt from the cached gridMET archive."""
    logger.info("autumn2025: fetching gridMET 2025 for %s cells", len(grid))
    gm = fetch_gridmet_for_grid(grid, 2025)
    gm["date"] = pd.to_datetime(gm["date"])
    gm = gm.merge(grid[["grid_id", "lat_center", "lon_center"]], on="grid_id", how="left")

    # Seasonal dryness normals (TerraClimate has no live feed), joined per calendar month.
    clim = pd.read_json(REFERENCE_DIR / "dryness_climatology.json")
    gm["month"] = gm["date"].dt.month
    gm = gm.merge(clim, on=["grid_id", "month"], how="left").drop(columns=["month"])
    for col in ["aet", "water_deficit"]:
        gm[col] = gm[col].fillna(gm[col].median())

    # Engineer over the FULL year so the rolling windows and the unbounded dry_streak
    # counter reach the same scale training saw — the live pipeline's ~26-day window is
    # exactly the corruption this panel exists to avoid.
    df = engineer_features(gm)
    df = merge_static_features(df)
    df["lightning_count"] = 0   # no GLM archive for 2025; it measures ~noise anyway

    fires = points_to_cell_days(pd.read_parquet(EVAL_DIR / "labels_2025.parquet"), grid)
    logger.info("autumn2025: %s unique (cell, day) ignitions Jul-Nov", f"{len(fires):,}")
    df = merge_recency(df, fires, warmup_days=150)

    lab = fires.assign(confirmed=1)
    df = df.merge(lab, on=["grid_id", "date"], how="left")
    df["confirmed"] = df["confirmed"].fillna(0).astype(int)
    df[TARGET_COL] = df["confirmed"]

    out = df[(df["date"] >= AUTUMN_START) & (df["date"] <= AUTUMN_END)].copy()
    logger.info("autumn2025: %s rows, %s cells, %s ignitions (base rate %.3f%%)",
                f"{len(out):,}", out.grid_id.nunique(), int(out.confirmed.sum()),
                100 * out.confirmed.mean())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    grid = pd.read_json(REFERENCE_DIR / "grid_cells.json")

    if not (EVAL_DIR / "panel_train201819.parquet").exists():
        train, hold = build_historical()
        train.to_parquet(EVAL_DIR / "panel_train201819.parquet", index=False)
        hold.to_parquet(EVAL_DIR / "panel_holdout2020.parquet", index=False)

    for name, builder in [("live2026", lambda: build_live2026(grid)),
                          ("autumn2025", lambda: build_autumn2025(grid))]:
        path = EVAL_DIR / f"panel_{name}.parquet"
        if path.exists():
            logger.info("%s already built, skipping", path.name)
            continue
        df = builder()
        df.to_parquet(path, index=False)
        logger.info("wrote %s (%.1f MB)", path.name, path.stat().st_size / 1e6)

    # Sanity: the recency block must actually vary, or the ablation's recency arm is a no-op.
    for name in ("holdout2020", "live2026", "autumn2025"):
        d = pd.read_parquet(EVAL_DIR / f"panel_{name}.parquet", columns=RECENCY_FEATURES)
        logger.info("%s recency: cell mean %.4f (%.1f%% nonzero) | nbr mean %.4f | days_since mean %.1f",
                    name, d.fire_recency_cell.mean(), 100 * (d.fire_recency_cell > 0).mean(),
                    d.fire_recency_nbr.mean(), d.days_since_fire_cell.mean())


if __name__ == "__main__":
    main()
