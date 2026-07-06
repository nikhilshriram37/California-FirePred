"""Fetch live inputs for daily scoring: recent gridMET + seasonal dryness.

- gridMET: the live weather/fire-danger backbone (20 of 28 features). We pull the
  current-year file (which contains everything up to the latest available day,
  ~1-4 day lag) and keep a trailing window long enough for the 14-day rolling
  features.
- TerraClimate aet / water_deficit: no real-time feed exists (monthly, lagged),
  so we use the per-cell seasonal normal for the target month from the historical
  dataset. Slow-moving, seasonal features — a defensible stand-in.

Lightning comes from :mod:`src.data_acquisition.fetch_glm` (GOES-GLM).
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from src.data_acquisition.config import PROCESSED_DIR, REFERENCE_DIR, REGIONS
from src.preprocessing.build_dataset import fetch_gridmet_for_grid

logger = logging.getLogger(__name__)

GRIDMET_VARS = ["tmmx", "rmin", "vs", "pr", "erc", "vpd", "fm100", "bi"]


def fetch_gridmet_recent(grid: pd.DataFrame, days: int = 21,
                         end_date: dt.date | None = None) -> pd.DataFrame:
    """Return the last ``days`` of gridMET for the grid (enough for 14d rollings).

    Pulls the current year, and the previous year too when the window straddles
    Jan 1, so the rolling features are fully populated at the window edge.
    """
    bbox = REGIONS["california"]
    year = (end_date or dt.date.today()).year

    # refresh_current=True: the current-year file grows daily, so always re-fetch
    # it (a stale cache would freeze the latest available day).
    frames = [fetch_gridmet_for_grid(grid, year, variables=GRIDMET_VARS, bbox=bbox,
                                     refresh_current=True)]
    cur = pd.to_datetime(frames[0]["date"])
    cutoff = pd.Timestamp(end_date) if end_date else cur.max()
    start = cutoff - pd.Timedelta(days=days + 5)
    if start.year < year:  # window crosses into last year — grab its tail too
        frames.insert(0, fetch_gridmet_for_grid(grid, year - 1, variables=GRIDMET_VARS, bbox=bbox))

    wx = pd.concat(frames, ignore_index=True)
    wx["date"] = pd.to_datetime(wx["date"])
    wx = wx[(wx["date"] >= start) & (wx["date"] <= cutoff)].copy()
    logger.info("gridMET: %s rows, %s..%s", f"{len(wx):,}",
                wx["date"].min().date(), wx["date"].max().date())
    return wx


def dryness_for_month(month: int, dataset_path=None) -> pd.DataFrame:
    """Per-cell seasonal normal aet / water_deficit for a calendar month.

    Computed from the historical dataset since TerraClimate has no live feed.
    Prefers the small committed reference file (works in the cloud); falls back to
    the full parquet locally.
    """
    ref = REFERENCE_DIR / "dryness_climatology.json"
    if ref.exists():
        clim = pd.read_json(ref)
        clim = clim[clim["month"] == month][["grid_id", "aet", "water_deficit"]].reset_index(drop=True)
        logger.info("dryness: seasonal normals for month=%s over %s cells (reference)", month, len(clim))
        return clim

    dataset_path = dataset_path or PROCESSED_DIR / "california_dataset.parquet"
    hist = pd.read_parquet(dataset_path, columns=["grid_id", "month", "aet", "water_deficit"])
    clim = (hist[hist["month"] == month]
            .groupby("grid_id")[["aet", "water_deficit"]].mean().reset_index())
    logger.info("dryness: seasonal normals for month=%s over %s cells", month, len(clim))
    return clim
