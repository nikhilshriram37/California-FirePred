"""Fire-danger index emulator: forecast weather -> erc / bi / fm100.

Standard weather forecasts give temp/humidity/wind/precip but NOT gridMET's
NFDRS fire-danger indices (erc, bi, fm100), which are among the risk model's top
features. This emulator reconstructs them autoregressively: each day's index is a
function of that day's weather plus the previous day's index value (the indices
have multi-day memory, exactly how NFDRS computes them). Validated to hold within
~7-17% of natural variation over a 7-day rollout (see project notes).

Used by the Route A forecast pipeline: seed from the last *observed* gridMET index,
then step forward through forecast days.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.data_acquisition.config import PROJECT_ROOT

EMU_DIR = PROJECT_ROOT / "models" / "emulator"

# Weather inputs available from a forecast (Open-Meteo/NWS); match gridMET names.
EMU_WEATHER = ["tmmx_c", "rmin", "vs", "pr", "vpd"]
EMU_TARGETS = ["erc", "bi", "fm100"]
EMU_FEATURES = EMU_WEATHER + [f"{t}_lag1" for t in EMU_TARGETS] + ["doy_cos"]


def build_emulator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag-1 index columns + doy_cos to a per-cell, date-sorted gridMET panel."""
    df = df.sort_values(["grid_id", "date"]).copy()
    g = df.groupby("grid_id")
    for t in EMU_TARGETS:
        df[f"{t}_lag1"] = g[t].shift(1)
    df["doy_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["date"]).dt.dayofyear / 365)
    return df


class Emulator:
    """Three regressors (erc/bi/fm100) + an autoregressive step()."""

    def __init__(self, models_dir: Path = EMU_DIR):
        self.models = {}
        for t in EMU_TARGETS:
            m = XGBRegressor()
            m.load_model(str(Path(models_dir) / f"{t}.json"))
            self.models[t] = m
        self.card = json.loads((Path(models_dir) / "emulator_card.json").read_text())

    def step(self, weather: dict, prev_index: dict, doy: int) -> dict:
        """Predict {erc,bi,fm100} for a day given its weather + previous day's index."""
        feat = ([weather[w] for w in EMU_WEATHER]
                + [prev_index[t] for t in EMU_TARGETS]
                + [np.cos(2 * np.pi * doy / 365)])
        x = np.array(feat, dtype=float).reshape(1, -1)
        return {t: float(self.models[t].predict(x)[0]) for t in EMU_TARGETS}

    def step_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Vectorized step over many cells. X must have the EMU_FEATURES columns."""
        return pd.DataFrame({t: self.models[t].predict(X[EMU_FEATURES]) for t in EMU_TARGETS},
                            index=X.index)


@lru_cache(maxsize=1)
def load_emulator(models_dir: str | None = None) -> Emulator:
    return Emulator(Path(models_dir) if models_dir else EMU_DIR)
