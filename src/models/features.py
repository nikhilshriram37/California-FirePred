"""Feature contract — the single source of truth for what the model consumes.

Training (``src/models/train.py``) and serving (``src/models/predict.py``,
``src/pipeline/score_daily.py``) all import :data:`FEATURE_COLS` from here so the
two paths can never silently drift. The derived features themselves are produced
by :func:`src.preprocessing.build_dataset.engineer_features`, which is reused
unchanged for live scoring — this module only fixes *which* columns, and in what
order, are handed to the model.
"""

from __future__ import annotations

import pandas as pd

# Identifiers + fire outcomes that must never be fed to the model.
ID_COLS: list[str] = ["grid_id", "date"]
TARGET_COL: str = "has_fire"
LEAK_COLS: list[str] = ["fire_count", "total_acres"]  # fire outcomes -> label leakage
DROP_COLS: list[str] = [*ID_COLS, TARGET_COL, *LEAK_COLS]

# The exact 28 model features, in a fixed, serialization-stable order. Mirrors the
# columns of data/processed/california_dataset.parquet with DROP_COLS removed —
# i.e. the same list the notebook built via ``[c for c in df.columns if c not in
# drop_cols]``. Order matters: XGBoost validates feature names/order at predict.
FEATURE_COLS: list[str] = [
    "rmin", "vs", "pr", "vpd", "fm100", "bi", "aet", "water_deficit",
    "lightning_count", "lat_center", "lon_center", "tmmx_c",
    "erc_7d", "erc_14d", "vpd_7d", "vpd_14d", "bi_7d", "bi_14d",
    "tmmx_7d", "rmin_7d", "dry_streak", "pr_7d", "pr_14d",
    "fm100_change_3d", "vpd_change_3d", "month", "month_sin", "doy_cos",
]


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
