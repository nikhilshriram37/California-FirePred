"""Train the fire-danger index emulator on full-year gridMET and save artifacts.

Trains erc/bi/fm100 ~ f(today weather + yesterday's index + doy_cos) on the CA grid
across the requested years (peak fire season included). Saves to models/emulator/.

Run:  python -m src.models.train_emulator                 # auto-detects full years
      python -m src.models.train_emulator --years 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.data_acquisition.config import RAW_DIR, REGIONS
from src.models.emulator import EMU_DIR, EMU_FEATURES, EMU_TARGETS, EMU_WEATHER, build_emulator_features
from src.preprocessing.build_dataset import fetch_gridmet_for_grid
from src.pipeline.score_daily import canonical_grid

logger = logging.getLogger(__name__)
RAW_VARS = ["tmmx", "rmin", "vs", "pr", "erc", "vpd", "fm100", "bi"]


def _complete_years() -> list[int]:
    """Years with all 8 gridMET variables cached."""
    cache = RAW_DIR / "gridmet"
    years = []
    for y in range(2018, datetime.now().year + 1):
        if all((cache / f"{v}_{y}.nc").exists() for v in RAW_VARS):
            years.append(y)
    return years


def train(years: list[int] | None = None, models_dir=EMU_DIR) -> dict:
    years = years or _complete_years()
    if not years:
        raise RuntimeError("no fully-cached gridMET years found in data/raw/gridmet")
    models_dir.mkdir(parents=True, exist_ok=True)
    grid = canonical_grid()

    logger.info("Loading gridMET years %s for CA cells ...", years)
    frames = []
    for y in years:
        wx = fetch_gridmet_for_grid(grid, y, variables=RAW_VARS, bbox=REGIONS["california"])
        frames.append(wx)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["tmmx_c"] = df["tmmx"] - 273.15
    df = df.dropna(subset=RAW_VARS).reset_index(drop=True)

    df = build_emulator_features(df).dropna(subset=[f"{t}_lag1" for t in EMU_TARGETS])
    logger.info("training rows: %s", f"{len(df):,}")

    # Hold out the latest 15% of dates to report honest accuracy.
    cut = df["date"].quantile(0.85)
    tr, te = df[df["date"] <= cut], df[df["date"] > cut]

    metrics = {}
    for t in EMU_TARGETS:
        m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
        m.fit(tr[EMU_FEATURES], tr[t])
        pred = m.predict(te[EMU_FEATURES])
        metrics[t] = {"r2": float(r2_score(te[t], pred)),
                      "mae": float(mean_absolute_error(te[t], pred))}
        m.save_model(str(models_dir / f"{t}.json"))
        logger.info("%s: R2=%.3f MAE=%.3f", t, metrics[t]["r2"], metrics[t]["mae"])

    card = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "years": years, "n_rows": int(len(df)),
        "weather_inputs": EMU_WEATHER, "targets": EMU_TARGETS,
        "metrics": metrics,
    }
    (models_dir / "emulator_card.json").write_text(json.dumps(card, indent=2))
    logger.info("saved emulator -> %s", models_dir)
    return card


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", help="years to train on")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(json.dumps(train(years=args.years), indent=2))
