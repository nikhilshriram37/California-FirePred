"""Phase-1 experiment: do topography + population features lift PR-AUC?

Compares the current feature set (BASE) against BASE + static features (EXT) under
two evaluations, WITHOUT touching the deployed model/contract:
  * temporal  — train 2018-19, test 2020 (matches deployment)
  * spatial CV — 5 spatial blocks (KMeans on cell centers); train on out-of-block
    cells, test on the held-out block. Honest spatial-generalization estimate.

Success = EXT PR-AUC > BASE PR-AUC (more recall at equal precision).
Run:  python -m scripts.experiment_static_features   (or: .venv/bin/python scripts/experiment_static_features.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.data_acquisition.config import PROCESSED_DIR, REFERENCE_DIR
from src.models.features import FEATURE_COLS, TARGET_COL
from src.models.train import TUNED_PARAMS

STATIC = ["elev_mean", "ruggedness", "slope_deg", "northness", "eastness", "log_pop"]


def _fit_predict(Xtr, ytr, Xte):
    m = XGBClassifier(**TUNED_PARAMS, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "california_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    static = pd.read_json(REFERENCE_DIR / "static_features.json")
    df = df.merge(static, on="grid_id", how="left")
    miss = df[STATIC].isna().any(axis=1).mean() * 100
    print(f"dataset {len(df):,} rows | static-feature NaN rows: {miss:.2f}%")

    base = FEATURE_COLS
    ext = FEATURE_COLS + STATIC
    y = df[TARGET_COL].to_numpy()

    # --- Temporal: train 2018-19, test 2020 ---
    tr = df[df["date"].dt.year.isin([2018, 2019])]
    te = df[df["date"].dt.year == 2020]
    print("\n=== TEMPORAL (train 2018-19 -> test 2020) ===")
    for name, cols in [("BASE (28)", base), ("EXT  (+static)", ext)]:
        p = _fit_predict(tr[cols], tr[TARGET_COL], te[cols])
        print(f"  {name:16s} PR-AUC = {average_precision_score(te[TARGET_COL], p):.4f}")

    # --- Spatial CV: 5 KMeans blocks on cell centers ---
    cells = df.drop_duplicates("grid_id")[["grid_id", "lat_center", "lon_center"]].copy()
    cells["fold"] = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(
        cells[["lat_center", "lon_center"]])
    fold = df.merge(cells[["grid_id", "fold"]], on="grid_id")["fold"].to_numpy()
    print("\n=== SPATIAL CV (5 blocks; aggregated out-of-fold PR-AUC) ===")
    for name, cols in [("BASE (28)", base), ("EXT  (+static)", ext)]:
        oof = np.zeros(len(df))
        for k in range(5):
            m = fold == k
            oof[m] = _fit_predict(df.loc[~m, cols], y[~m], df.loc[m, cols])
        print(f"  {name:16s} PR-AUC = {average_precision_score(y, oof):.4f}")

    print("\nBaseline (deployed) held-out 2020 PR-AUC was ~0.069.")


if __name__ == "__main__":
    main()
