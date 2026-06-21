"""Close the prediction -> outcome loop: backfill has_fire from fused fire sources.

For each recent prediction day, determine where fires actually occurred and set
feature_history.has_fire, fusing three sources by their strengths:

  * IRWIN / WFIGS  — interagency confirmed incidents (discovery date, location,
                     cause, size). FPA-FOD-aligned ground truth; primary positive.
  * CAL FIRE       — California official incidents; CA-specific confirmation.
  * NASA FIRMS     — satellite active-fire detections; supplementary recall
                     (catches small/unreported fires the agencies don't log).

A cell is has_fire=1 if a confirmed incident started there OR FIRMS detected fire.
label_source records which source(s) confirmed it ('irwin', 'calfire', 'firms',
or '+'-joined) so retraining can prefer high-fidelity labels. Re-labels a trailing
window each run to absorb late-arriving incidents/detections. FIRMS detections are
also archived to active_fires.

Run:  python -m src.pipeline.backfill_labels                 # last 7 days
      python -m src.pipeline.backfill_labels --days 10
      python -m src.pipeline.backfill_labels --date 2026-06-18
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import urllib.parse

import numpy as np
import pandas as pd
import requests

from src.data_acquisition.config import NASA_FIRMS_MAP_KEY, REFERENCE_DIR, REGIONS
from src.pipeline.supabase_io import get_client

logger = logging.getLogger(__name__)

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"]
IRWIN_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
             "WFIGS_Incident_Locations_YearToDate/FeatureServer/0/query")
CALFIRE_URL = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List"
CELL_DEG = 0.1
HALF = CELL_DEG / 2


def _grid() -> pd.DataFrame:
    """Canonical grid from the small committed reference file (no heavy deps)."""
    return pd.read_json(REFERENCE_DIR / "grid_cells.json")


# --------------------------------------------------------------------------- #
# Source fetchers — each returns a DataFrame with latitude, longitude, date
# --------------------------------------------------------------------------- #

def fetch_firms_for_date(date: dt.date, map_key: str | None = None) -> pd.DataFrame:
    """CA FIRMS detections for one date, across satellites (+ raw cols for archive)."""
    map_key = map_key or NASA_FIRMS_MAP_KEY
    b = REGIONS["california"]
    bbox = f"{b['west']},{b['south']},{b['east']},{b['north']}"
    ds = date.strftime("%Y-%m-%d")
    frames = []
    for src in FIRMS_SOURCES:
        try:
            r = requests.get(f"{FIRMS_BASE}/{map_key}/{src}/{bbox}/1/{ds}", timeout=120)
            if r.ok:
                df = pd.read_csv(io.StringIO(r.text))
                if not df.empty and {"latitude", "longitude"} <= set(df.columns):
                    df["satellite"] = df.get("satellite", src)
                    frames.append(df)
        except Exception as e:
            logger.warning("FIRMS %s %s failed: %s", src, ds, e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["latitude", "longitude"])


def fetch_irwin_ca() -> pd.DataFrame:
    """All CA interagency (IRWIN/WFIGS) incidents year-to-date: lat, lon, discovery date."""
    rows, offset = [], 0
    while True:
        q = {"where": "POOState='US-CA'",
             "outFields": "FireDiscoveryDateTime,InitialLatitude,InitialLongitude",
             "resultOffset": offset, "resultRecordCount": 2000, "f": "json"}
        try:
            d = requests.get(f"{IRWIN_URL}?{urllib.parse.urlencode(q)}", timeout=90).json()
        except Exception as e:
            logger.warning("IRWIN fetch failed: %s", e); break
        feats = d.get("features", [])
        for f in feats:
            a = f["attributes"]
            ts, la, lo = a.get("FireDiscoveryDateTime"), a.get("InitialLatitude"), a.get("InitialLongitude")
            if ts and la and lo:
                rows.append({"latitude": la, "longitude": lo,
                             "date": dt.datetime.fromtimestamp(ts / 1000, dt.UTC).date()})
        if len(feats) < 2000:
            break
        offset += 2000
    df = pd.DataFrame(rows)
    logger.info("IRWIN: %d CA incidents YTD", len(df))
    return df


def fetch_calfire_ca() -> pd.DataFrame:
    """CA official incidents (CAL FIRE) for the year: lat, lon, start date."""
    try:
        data = requests.get(f"{CALFIRE_URL}?inactive=true&year={dt.date.today().year}", timeout=90).json()
    except Exception as e:
        logger.warning("CAL FIRE fetch failed: %s", e); return pd.DataFrame()
    rows = data if isinstance(data, list) else data.get("Incidents", [])
    out = []
    for r in rows:
        la, lo, st = r.get("Latitude"), r.get("Longitude"), r.get("Started")
        if la and lo and st:
            try:
                out.append({"latitude": float(la), "longitude": float(lo),
                            "date": pd.to_datetime(st).date()})
            except Exception:
                pass
    df = pd.DataFrame(out)
    logger.info("CAL FIRE: %d CA incidents this year", len(df))
    return df


# --------------------------------------------------------------------------- #
# Mapping + persistence
# --------------------------------------------------------------------------- #

def points_to_grid_ids(pts: pd.DataFrame, grid: pd.DataFrame) -> set[int]:
    """Map (latitude, longitude) points to grid cells via the training flooring rule."""
    if pts is None or pts.empty:
        return set()
    lookup = {(round(r.lat_center, 2), round(r.lon_center, 2)): int(r.grid_id) for r in grid.itertuples()}
    ids: set[int] = set()
    for lat, lon in zip(pts["latitude"].to_numpy(), pts["longitude"].to_numpy()):
        clat = round(np.floor(lat / CELL_DEG) * CELL_DEG + HALF, 2)
        clon = round(np.floor(lon / CELL_DEG) * CELL_DEG + HALF, 2)
        gid = lookup.get((clat, clon))
        if gid is not None:
            ids.add(gid)
    return ids


def _archive_fires(client, det: pd.DataFrame, ds: str) -> None:
    client.table("active_fires").delete().eq("acq_date", ds).execute()
    if det.empty:
        return
    rows = [{
        "latitude": float(r.latitude), "longitude": float(r.longitude),
        "frp": float(getattr(r, "frp")) if pd.notna(getattr(r, "frp", None)) else None,
        "confidence": str(getattr(r, "confidence", "")) or None, "acq_date": ds,
        "acq_time": str(getattr(r, "acq_time", "")) or None,
        "satellite": str(getattr(r, "satellite", "")) or None,
    } for r in det.itertuples()]
    for i in range(0, len(rows), 1000):
        client.table("active_fires").insert(rows[i:i + 1000]).execute()


def _has_label_source(client) -> bool:
    try:
        client.table("feature_history").select("label_source").limit(1).execute()
        return True
    except Exception:
        logger.info("label_source column absent — run migration 0002 to enable source tracking")
        return False


def backfill_date(client, grid, date, irwin, calfire, with_source: bool) -> dict:
    """Label one date's feature_history rows from fused sources; archive FIRMS."""
    ds = date.strftime("%Y-%m-%d")
    firms = fetch_firms_for_date(date)

    irwin_ids = points_to_grid_ids(irwin[irwin["date"] == date] if not irwin.empty else irwin, grid)
    calfire_ids = points_to_grid_ids(calfire[calfire["date"] == date] if not calfire.empty else calfire, grid)
    firms_ids = points_to_grid_ids(firms, grid)
    all_ids = irwin_ids | calfire_ids | firms_ids

    # reset the day, then write fire cells grouped by their source combo
    client.table("feature_history").update(
        {"has_fire": 0, **({"label_source": None} if with_source else {})}).eq("date", ds).execute()

    by_source: dict[str, list[int]] = {}
    for gid in all_ids:
        src = "+".join(s for s, ids in
                       (("irwin", irwin_ids), ("calfire", calfire_ids), ("firms", firms_ids)) if gid in ids)
        by_source.setdefault(src, []).append(gid)
    for src, ids in by_source.items():
        payload = {"has_fire": 1, **({"label_source": src} if with_source else {})}
        for i in range(0, len(ids), 500):
            client.table("feature_history").update(payload).eq("date", ds).in_("grid_id", ids[i:i + 500]).execute()

    _archive_fires(client, firms, ds)
    confirmed = len(irwin_ids | calfire_ids)
    logger.info("%s: fire cells=%d (confirmed=%d, firms-only=%d)",
                ds, len(all_ids), confirmed, len(all_ids - irwin_ids - calfire_ids))
    return {"date": ds, "fire_cells": len(all_ids), "confirmed": confirmed,
            "firms_only": len(all_ids - irwin_ids - calfire_ids)}


def backfill_range(days: int = 7, end: dt.date | None = None) -> list[dict]:
    client = get_client()
    if client is None:
        logger.warning("Supabase not configured — nothing to backfill")
        return []
    grid = _grid()
    irwin, calfire = fetch_irwin_ca(), fetch_calfire_ca()  # fetched once per run
    ws = _has_label_source(client)
    end = end or dt.date.today()
    return [backfill_date(client, grid, end - dt.timedelta(days=d), irwin, calfire, ws)
            for d in range(1, days + 1)]


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
        d = dt.date.fromisoformat(args.date)
        print(backfill_date(client, _grid(), d, fetch_irwin_ca(), fetch_calfire_ca(), _has_label_source(client)))
    else:
        for r in backfill_range(days=args.days):
            print(r)


if __name__ == "__main__":
    main()
