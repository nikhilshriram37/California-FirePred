"""Close the prediction -> outcome loop: backfill has_fire + archive FIRMS fires.

For each recent prediction day, query NASA FIRMS for fires that actually occurred,
map detections to the model's grid cells, and set feature_history.has_fire (1 where
a fire was detected that day, 0 otherwise). Re-checks a trailing window each run
because FIRMS NRT detections arrive and get corrected over a few days. Also archives
the raw detections to active_fires so we keep a historical fire record.

This is what makes weekly/biweekly retraining meaningful: feature_history accumulates
features + prediction (already stored) and now the realized outcome.

Label note: the original model trained on FPA-FOD official ignition records; this
operational loop uses FIRMS active-fire detections (active heat) as the near-real-time
label. Slightly different definition — fine for monitoring + retraining; reconcile
with CAL FIRE / NIFC incident records later for higher-fidelity labels.

Run:  python -m src.pipeline.backfill_labels                # last 7 days
      python -m src.pipeline.backfill_labels --days 10
      python -m src.pipeline.backfill_labels --date 2026-06-18
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging

import numpy as np
import pandas as pd
import requests

from src.data_acquisition.config import NASA_FIRMS_MAP_KEY, REFERENCE_DIR, REGIONS
from src.pipeline.supabase_io import get_client

logger = logging.getLogger(__name__)

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# Same 3 satellites the dashboard shows, for fuller coverage.
SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"]
CELL_DEG = 0.1
HALF = CELL_DEG / 2


def _grid() -> pd.DataFrame:
    """Canonical grid from the small committed reference file (no heavy deps)."""
    return pd.read_json(REFERENCE_DIR / "grid_cells.json")


def fetch_firms_for_date(date: dt.date, map_key: str | None = None) -> pd.DataFrame:
    """All CA FIRMS detections for one calendar date, across satellites."""
    map_key = map_key or NASA_FIRMS_MAP_KEY
    b = REGIONS["california"]
    bbox = f"{b['west']},{b['south']},{b['east']},{b['north']}"
    ds = date.strftime("%Y-%m-%d")
    frames = []
    for src in SOURCES:
        url = f"{FIRMS_BASE}/{map_key}/{src}/{bbox}/1/{ds}"
        try:
            r = requests.get(url, timeout=120)
            if not r.ok:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if not df.empty and {"latitude", "longitude"} <= set(df.columns):
                df["satellite"] = df.get("satellite", src)
                frames.append(df)
        except Exception as e:
            logger.warning("FIRMS %s %s failed: %s", src, ds, e)
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude"])
    return pd.concat(frames, ignore_index=True)


def detections_to_grid_ids(det: pd.DataFrame, grid: pd.DataFrame) -> set[int]:
    """Map detections to grid cells using the same flooring as the training pipeline.

    A detection at (lat, lon) belongs to the cell whose center is
    (floor(lon/0.1)*0.1 + 0.05, floor(lat/0.1)*0.1 + 0.05).
    """
    if det.empty:
        return set()
    lookup = {
        (round(r.lat_center, 2), round(r.lon_center, 2)): int(r.grid_id)
        for r in grid.itertuples()
    }
    ids: set[int] = set()
    for lat, lon in zip(det["latitude"].to_numpy(), det["longitude"].to_numpy()):
        clat = round(np.floor(lat / CELL_DEG) * CELL_DEG + HALF, 2)
        clon = round(np.floor(lon / CELL_DEG) * CELL_DEG + HALF, 2)
        gid = lookup.get((clat, clon))
        if gid is not None:
            ids.add(gid)
    return ids


def _archive_fires(client, det: pd.DataFrame, ds: str) -> None:
    """Replace active_fires rows for the date with the current detections."""
    client.table("active_fires").delete().eq("acq_date", ds).execute()
    if det.empty:
        return
    rows = []
    for r in det.itertuples():
        rows.append({
            "latitude": float(r.latitude), "longitude": float(r.longitude),
            "frp": float(getattr(r, "frp", float("nan"))) if pd.notna(getattr(r, "frp", None)) else None,
            "confidence": str(getattr(r, "confidence", "")) or None,
            "acq_date": ds,
            "acq_time": str(getattr(r, "acq_time", "")) or None,
            "satellite": str(getattr(r, "satellite", "")) or None,
        })
    for i in range(0, len(rows), 1000):
        client.table("active_fires").insert(rows[i:i + 1000]).execute()


def backfill_date(client, grid: pd.DataFrame, date: dt.date) -> dict:
    """Label one date's feature_history rows and archive that day's fires."""
    ds = date.strftime("%Y-%m-%d")
    det = fetch_firms_for_date(date)
    fire_ids = detections_to_grid_ids(det, grid)

    # Set the whole day's scored cells to 0, then flip detected cells to 1.
    client.table("feature_history").update({"has_fire": 0}).eq("date", ds).execute()
    if fire_ids:
        ids = list(fire_ids)
        for i in range(0, len(ids), 500):
            client.table("feature_history").update({"has_fire": 1}) \
                .eq("date", ds).in_("grid_id", ids[i:i + 500]).execute()

    _archive_fires(client, det, ds)
    logger.info("%s: %d detections -> %d fire cells labeled", ds, len(det), len(fire_ids))
    return {"date": ds, "detections": int(len(det)), "fire_cells": len(fire_ids)}


def backfill_range(days: int = 7, end: dt.date | None = None) -> list[dict]:
    client = get_client()
    if client is None:
        logger.warning("Supabase not configured — nothing to backfill")
        return []
    grid = _grid()
    end = end or dt.date.today()
    out = []
    for d in range(1, days + 1):  # yesterday back to `days` ago (today isn't complete)
        out.append(backfill_date(client, grid, end - dt.timedelta(days=d)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="trailing days to (re)label")
    ap.add_argument("--date", help="label a single YYYY-MM-DD date instead")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.date:
        client = get_client()
        if client is None:
            print("Supabase not configured"); return
        print(backfill_date(client, _grid(), dt.date.fromisoformat(args.date)))
    else:
        for r in backfill_range(days=args.days):
            print(r)


if __name__ == "__main__":
    main()
