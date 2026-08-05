"""Feature contract — the single source of truth for what the model consumes.

Training (``src/models/train.py``) and serving (``src/models/predict.py``,
``src/pipeline/score_daily.py``) all import :data:`FEATURE_COLS` from here so the
two paths can never silently drift. The derived features themselves are produced
by :func:`src.preprocessing.build_dataset.engineer_features`, which is reused
unchanged for live scoring — this module only fixes *which* columns, and in what
order, are handed to the model.
"""

import pandas as pd

from src.data_acquisition.config import REFERENCE_DIR
from src.models.recency import RECENCY_FEATURES

# Identifiers + fire outcomes that must never be fed to the model.
ID_COLS: list[str] = ["grid_id", "date"]
TARGET_COL: str = "has_fire"
LEAK_COLS: list[str] = ["fire_count", "total_acres"]  # fire outcomes -> label leakage
DROP_COLS: list[str] = [*ID_COLS, TARGET_COL, *LEAK_COLS]

# Static, per-cell features (constant over time): topography + human exposure.
# Sourced offline into data/reference/static_features.json (scripts/build_*.py) and
# merged by grid_id in both training and serving via merge_static_features().
STATIC_FEATURES: list[str] = [
    "elev_mean", "ruggedness", "slope_deg", "northness", "eastness", "log_pop",
]

# The full model feature set, in a fixed, serialization-stable order. The first 27
# are the weather/fuel/temporal features engineered by engineer_features; the static
# block is appended, then the recency block. Order matters: XGBoost validates feature
# names/order at predict.
#
# `lightning_count` was removed 2026-08-05. The feature-block ablation
# (docs/feature_ablation_2026-08-05.md) measured its marginal contribution on four
# evaluation panels and found a coin flip every time (|delta PR-AUC| <= 0.0003, positive
# in ~50% of matched pairs). It was near-constant zero in the NOAA Storm Events training
# data, so the model never learned from it, and the live GOES-GLM values it now receives
# are meaningless to it. Still fetched and stored in feature_history for analysis; simply
# no longer an input.
#
# The recency block carries a *current* spatial prior — see src/models/recency.py.
# Without it the model's only sense of "where fires happen" was whatever 2018-2020
# looked like, frozen into lat/lon; measured live, that stale prior cost ~4x PR-AUC.
# It is produced by merge_recency() in both training and serving, never by
# engineer_features, because it needs the label history rather than the weather feed.
FEATURE_COLS: list[str] = [
    "rmin", "vs", "pr", "vpd", "fm100", "bi", "aet", "water_deficit",
    "lat_center", "lon_center", "tmmx_c",
    "erc_7d", "erc_14d", "vpd_7d", "vpd_14d", "bi_7d", "bi_14d",
    "tmmx_7d", "rmin_7d", "dry_streak", "pr_7d", "pr_14d",
    "fm100_change_3d", "vpd_change_3d", "month", "month_sin", "doy_cos",
    *STATIC_FEATURES,
    *RECENCY_FEATURES,
]


def merge_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the static per-cell features onto a frame keyed by grid_id."""
    static = pd.read_json(REFERENCE_DIR / "static_features.json")
    keep = ["grid_id", *[c for c in STATIC_FEATURES if c in static.columns]]
    return df.merge(static[keep], on="grid_id", how="left")


def select_features(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Return a DataFrame with exactly :data:`FEATURE_COLS`, in order.

    Args:
        df: Any frame containing (at least) the engineered feature columns.
        strict: If ``True``, raise when a feature is missing. If ``False``, the
            missing column is filled with ``NaN`` — XGBoost handles NaN natively,
            which is the right behaviour for a live feed that briefly drops a
            variable.
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing and strict:
        raise ValueError(f"missing required feature columns: {missing}")
    return df.reindex(columns=FEATURE_COLS)
