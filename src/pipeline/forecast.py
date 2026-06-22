"""Route A: 1-5 day wildfire risk FORECAST.

Pipeline:
  1. recent observed gridMET  -> seeds rolling windows + last real index state
  2. Open-Meteo forecast weather (bridges gridMET's lag, extends ahead)
  3. autoregressive emulator   -> reconstruct erc/bi/fm100 for each forecast day
  4. engineer_features on the continuous (observed + forecast) panel
  5. existing risk model       -> tier per cell, per target day (today+1 .. today+H)

Forecast inputs the model didn't train to forecast: lightning has no forecast feed
(set 0); aet/water_deficit use the seasonal normal (slow-moving). Honest useful
horizon ~1-5 days; skill decays with the weather forecast.

Run:  python -m src.pipeline.forecast                # writes per-horizon GeoJSON
      python -m src.pipeline.forecast --horizons 5 --no-persist
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PACIFIC = ZoneInfo("America/Los_Angeles")

from src.data_acquisition.fetch_forecast import fetch_forecast
from src.data_acquisition.fetch_live import dryness_for_month, fetch_gridmet_recent
from src.models.emulator import EMU_TARGETS, load_emulator
from src.models.predict import load_model
from src.pipeline.build_live_features import build_live_features
from src.pipeline.score_daily import canonical_grid
from src.pipeline.snapshot import SNAPSHOT_DIR, build_meta, day_to_feature_collection, write_snapshot

logger = logging.getLogger(__name__)
RAW_VARS = ["tmmx", "rmin", "vs", "pr", "erc", "vpd", "fm100", "bi"]


def _emulate_forecast_days(grid, obs, fc, fdays):
    """Autoregressively reconstruct erc/bi/fm100 for each forecast day (vectorized)."""
    emu = load_emulator()
    d_obs = obs["date"].max()
    state = (obs[obs["date"] == d_obs].set_index("grid_id")[EMU_TARGETS])  # seed = last real
    rows = []
    for d in fdays:
        w = fc[fc["date"] == d].set_index("grid_id")
        cells = state.index.intersection(w.index)
        X = pd.DataFrame(index=cells)
        for c in ["tmmx_c", "rmin", "vs", "pr", "vpd"]:
            X[c] = w.loc[cells, c]
        for t in EMU_TARGETS:
            X[f"{t}_lag1"] = state.loc[cells, t]
        X["doy_cos"] = np.cos(2 * np.pi * pd.Timestamp(d).dayofyear / 365)
        pred = emu.step_batch(X)            # erc/bi/fm100 for this day
        state = pred                         # chain forward
        day = w.loc[cells].reset_index()
        day["date"] = pd.Timestamp(d)
        for t in EMU_TARGETS:
            day[t] = pred[t].values
        day["tmmx"] = day["tmmx_c"] + 273.15  # back to K for engineer_features schema
        rows.append(day)
    return pd.concat(rows, ignore_index=True)


def run_forecast(horizons: int = 5, persist: bool = True, write: bool = True) -> list[dict]:
    grid = canonical_grid()
    # "Today" is the California (Pacific) calendar day, so dates match the audience.
    pac_today = datetime.now(PACIFIC).date()

    # 1. observed gridMET (raw 8 vars) up to the latest published day
    obs = fetch_gridmet_recent(grid, days=21)
    obs["date"] = pd.to_datetime(obs["date"])
    d_obs = obs["date"].max()

    # 2. forecast weather (Pacific days), bridging the gap from d_obs to today+H
    gap = max(0, (pac_today - d_obs.date()).days)
    fc = fetch_forecast(grid, days=horizons + 1, past_days=max(5, gap + 2))
    fc["date"] = pd.to_datetime(fc["date"])

    # Contiguous targets: today (h=0) through +H, all real Pacific dates.
    target_dates = [pd.Timestamp(pac_today + timedelta(days=h)) for h in range(0, horizons + 1)]
    fdays = sorted(d for d in fc["date"].unique() if d > d_obs and d <= max(target_dates))
    logger.info("observed thru %s; pac_today=%s; emulating %d days -> targets %s..%s",
                d_obs.date(), pac_today, len(fdays), target_dates[0].date(), target_dates[-1].date())

    # 3. emulate indices for forecast days, then stitch a continuous raw panel
    fdf = _emulate_forecast_days(grid, obs, fc, fdays)
    panel = pd.concat([obs[["grid_id", "date", *RAW_VARS]], fdf[["grid_id", "date", *RAW_VARS]]],
                      ignore_index=True)

    # 4. features for the whole panel (rolling windows span observed + forecast)
    dryness = dryness_for_month(int(target_dates[0].month))
    lightning_zero = pd.DataFrame({"grid_id": grid["grid_id"], "lightning_count": 0})

    model = load_model()
    results = []
    for h, td in enumerate(target_dates):  # h = 0..horizons
        day, _ = build_live_features(grid, panel, dryness, lightning_zero, target_date=td)
        day = day.join(model.predict(day))
        geojson = day_to_feature_collection(day)
        meta = build_meta(day, td.strftime("%Y-%m-%d"), mode="forecast",
                          source="forecast: Open-Meteo + emulated fire-danger",
                          horizon=h, run_date=pac_today.isoformat())
        if write:
            write_snapshot(geojson, meta, SNAPSHOT_DIR / "forecast" / f"h{h}")
        if persist:
            from src.pipeline.supabase_io import persist_forecast
            persist_forecast(day, meta)
        counts = day["tier"].value_counts().reindex(["Red", "Yellow", "Green"]).fillna(0).astype(int)
        logger.info("h+%d %s: %s", h, td.date(), counts.to_dict())
        results.append(meta)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, default=5)
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for m in run_forecast(horizons=args.horizons, persist=not args.no_persist, write=not args.no_write):
        print(m["data_date"], m["tier_counts"])


if __name__ == "__main__":
    main()
