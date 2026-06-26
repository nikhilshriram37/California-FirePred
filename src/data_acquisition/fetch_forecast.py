"""Forecast weather for the CA grid via Open-Meteo (free, no key, ~7-day daily).

Returns the forecast-available subset of the model's raw inputs, named to match
gridMET: tmmx_c (degC), rmin (% min RH), vs (m/s), pr (mm), vpd (kPa). The
fire-danger indices (erc/bi/fm100) are NOT forecast here — they're reconstructed
by the emulator (src/models/emulator.py) in the forecast pipeline.

Open-Meteo accepts many locations per request, so we batch the grid cells.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
DAILY = ["temperature_2m_max", "relative_humidity_2m_min", "wind_speed_10m_max",
         "precipitation_sum", "vapour_pressure_deficit_max"]
# Open-Meteo daily field -> gridMET-style model column
RENAME = {
    "temperature_2m_max": "tmmx_c",
    "relative_humidity_2m_min": "rmin",
    "wind_speed_10m_max": "vs",
    "precipitation_sum": "pr",
    "vapour_pressure_deficit_max": "vpd",
}
BATCH = 150
# Forecast-grid resolution. Coarser than the risk grid on purpose: Open-Meteo is
# flaky from CI, and 0.5deg cuts the request count ~4x (the biggest reliability
# lever). The underlying forecast model is ~25km native, so a multi-day forecast
# loses little to ~50km sampling.
COARSE = 0.5


def _fetch_points(lats: list[float], lons: list[float], days: int, past_days: int, tz: str) -> list[dict]:
    """Fetch Open-Meteo daily forecast for points, batched + throttled with retry."""
    locs: list[dict] = []
    for s in range(0, len(lats), BATCH):
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in lats[s:s + BATCH]),
            "longitude": ",".join(f"{v:.4f}" for v in lons[s:s + BATCH]),
            "daily": ",".join(DAILY), "past_days": past_days,
            "forecast_days": days, "timezone": tz, "wind_speed_unit": "ms",
        }
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                # (connect, read) timeout: generous connect for Open-Meteo's slow
                # TLS handshakes from CI; timeouts are retried (not just 429s).
                r = requests.get(OPEN_METEO, params=params, timeout=(30, 60))
                if r.status_code == 429:
                    time.sleep(8 * (attempt + 1))
                    continue
                r.raise_for_status()
                payload = r.json()
                locs.extend(payload if isinstance(payload, list) else [payload])
                break
            except requests.exceptions.RequestException as e:
                last_err = e  # timeout / connection reset / transient 5xx -> back off
                logger.warning("Open-Meteo batch retry %d: %s", attempt + 1, str(e)[:80])
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Open-Meteo failed after retries: {last_err}")
        time.sleep(1)
    return locs


def fetch_forecast(grid: pd.DataFrame, days: int = 7, past_days: int = 0,
                   tz: str = "America/Los_Angeles") -> pd.DataFrame:
    """Return a tidy forecast panel: grid_id, date, tmmx_c, rmin, vs, pr, vpd.

    Daily values are aggregated to ``tz`` calendar days (Pacific by default, so
    dates align to the California audience). Cells are mapped to a coarse
    0.25-degree forecast grid (each distinct coarse point fetched once) to stay
    within Open-Meteo's rate limits. ``past_days`` also returns recent days (used
    to bridge gridMET's publishing lag).
    """
    cells = grid[["grid_id", "lat_center", "lon_center"]].copy()
    cells["clat"] = (np.round(cells["lat_center"] / COARSE) * COARSE).round(4)
    cells["clon"] = (np.round(cells["lon_center"] / COARSE) * COARSE).round(4)
    pts = cells[["clat", "clon"]].drop_duplicates().reset_index(drop=True)
    logger.info("forecast: %s cells -> %s coarse points", len(cells), len(pts))

    locs = _fetch_points(pts["clat"].tolist(), pts["clon"].tolist(), days, past_days, tz)

    # Build per-coarse-point frames, keyed by (clat, clon).
    by_pt = {}
    for (clat, clon), loc in zip(zip(pts["clat"], pts["clon"]), locs):
        d = loc["daily"]
        sub = pd.DataFrame({"date": pd.to_datetime(d["time"])})
        for k in DAILY:
            sub[RENAME[k]] = d[k]
        by_pt[(clat, clon)] = sub

    out = []
    for r in cells.itertuples():
        sub = by_pt.get((r.clat, r.clon))
        if sub is None:
            continue
        s = sub.copy()
        s["grid_id"] = int(r.grid_id)
        out.append(s)
    df = pd.concat(out, ignore_index=True)
    logger.info("forecast: %s cell-days, %s..%s, %s cells",
                f"{len(df):,}", df["date"].min().date(), df["date"].max().date(),
                df["grid_id"].nunique())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.pipeline.score_daily import canonical_grid
    fc = fetch_forecast(canonical_grid().head(20))
    print(fc.head(8).to_string())
