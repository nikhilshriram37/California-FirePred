"""Does an explicit *decaying* fire-recency feature lift PR-AUC?

Motivation. The retrain track's memorisation guard showed the booster gaining +133% on
cells that burned during live training and +9% on cells that did not. That signal is
real — historically, same-cell repeat ignitions have a median gap of 15 days and 35%
recur within a week — but the model has no feature carrying it, so it smuggles it in
through lat/lon, where it never decays. A cell that burned in July stays hot forever.

The fix under test: give the model the signal explicitly, in a form that *decays*.

Causality. For a row (cell c, day t) every recency feature is built from fires strictly
before t, via the recursion ``r[t] = r[t-1] * exp(-1/tau) + fire[t-1]``. Nothing from
day t itself enters, so this is servable: at scoring time yesterday's labels are known.

The honest test is not the headline number. A recency feature can inflate overall PR-AUC
purely by predicting "it is still burning where it is burning", which the dashboard's
FIRMS overlay already shows. So this script also reports PR-AUC restricted to cells with
*no* recent fire — the new-ignition case, which is what the product is actually for —
and a spatial CV, to confirm the gain is not just a fancier way to memorise location.

Run:  .venv/bin/python scripts/experiment_recency_features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.data_acquisition.config import PROCESSED_DIR
from src.models.features import FEATURE_COLS, TARGET_COL, merge_static_features
from src.models.train import TUNED_PARAMS, TRAIN_YEARS, TEST_YEAR

TAU_DAYS = 14.0        # decay half-ish life; ~ the 15-day median repeat-ignition gap
DAYS_SINCE_CAP = 365   # cells that never burned get the cap, not NaN

CELL_FEATS = ["fire_recency_cell", "days_since_fire_cell"]
NBR_FEATS = ["fire_recency_nbr"]


def build_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Attach causal, decaying fire-recency features to a complete cell x day panel."""
    piv = df.pivot(index="date", columns="grid_id", values=TARGET_COL)
    piv = piv.sort_index()
    dates, cells = piv.index.to_numpy(), piv.columns.to_numpy()
    M = piv.to_numpy(dtype=np.float32)          # (days, cells), 1 = burned that day
    n_d, n_c = M.shape

    # --- neighbour fire counts (3x3 on the 0.1-degree grid) ---
    cen = df.drop_duplicates("grid_id").set_index("grid_id").loc[cells]
    ix = np.round(cen["lon_center"].to_numpy() / 0.1).astype(int)
    iy = np.round(cen["lat_center"].to_numpy() / 0.1).astype(int)
    pos = {(x, y): j for j, (x, y) in enumerate(zip(ix, iy))}
    nbr_of = [[pos[(x + dx, y + dy)]
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if (dx or dy) and (x + dx, y + dy) in pos]
              for x, y in zip(ix, iy)]
    N = np.zeros_like(M)
    for j, nb in enumerate(nbr_of):
        if nb:
            N[:, j] = M[:, nb].sum(axis=1)

    # --- causal decay: r[t] uses only rows strictly before t ---
    decay = float(np.exp(-1.0 / TAU_DAYS))
    R = np.zeros_like(M)
    RN = np.zeros_like(M)
    DS = np.full((n_d, n_c), float(DAYS_SINCE_CAP), dtype=np.float32)
    for t in range(1, n_d):
        R[t] = R[t - 1] * decay + M[t - 1]
        RN[t] = RN[t - 1] * decay + N[t - 1]
        burned = M[t - 1] > 0
        DS[t] = np.where(burned, 1.0, np.minimum(DS[t - 1] + 1.0, DAYS_SINCE_CAP))

    long = pd.DataFrame({
        "date": np.repeat(dates, n_c),
        "grid_id": np.tile(cells, n_d),
        "fire_recency_cell": R.ravel(),
        "fire_recency_nbr": RN.ravel(),
        "days_since_fire_cell": DS.ravel(),
    })
    return df.merge(long, on=["date", "grid_id"], how="left")


def _fit_predict(tr, te, cols):
    m = XGBClassifier(**TUNED_PARAMS, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42)
    m.fit(tr[cols], tr[TARGET_COL])
    return m.predict_proba(te[cols])[:, 1]


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "california_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = merge_static_features(df)
    print(f"panel: {len(df):,} rows | {df.grid_id.nunique()} cells x {df.date.nunique()} days")

    df = build_recency(df)
    print(f"recency built. leakage check — corr(fire_recency_cell, has_fire) computed on "
          f"strictly-past data only\n")

    tr = df[df["date"].dt.year.isin(TRAIN_YEARS)]
    te = df[df["date"].dt.year == TEST_YEAR].copy()
    yte = te[TARGET_COL].to_numpy()

    variants = {
        "BASE (34)": FEATURE_COLS,
        "+cell recency": FEATURE_COLS + CELL_FEATS,
        "+cell+nbr recency": FEATURE_COLS + CELL_FEATS + NBR_FEATS,
    }

    print("=" * 74)
    print(f"TEMPORAL — train {TRAIN_YEARS} -> test {TEST_YEAR}")
    print("=" * 74)
    preds = {}
    for name, cols in variants.items():
        p = _fit_predict(tr, te, cols)
        preds[name] = p
        print(f"  {name:20s} PR-AUC {average_precision_score(yte, p):.5f}   "
              f"ROC {roc_auc_score(yte, p):.5f}")

    # --- the honest split: does it help where nothing burned recently? ---
    print()
    print("=" * 74)
    print("IS IT PREDICTION OR PERSISTENCE?")
    print("=" * 74)
    quiet = te["fire_recency_cell"].to_numpy() < 0.01   # no fire in this cell recently
    print(f"  {TEST_YEAR} rows with no recent fire in-cell: {quiet.sum():,} "
          f"({quiet.mean():.1%}) holding {int(yte[quiet].sum()):,} of {int(yte.sum()):,} fires")
    for name, p in preds.items():
        q = average_precision_score(yte[quiet], p[quiet])
        h = (average_precision_score(yte[~quiet], p[~quiet])
             if yte[~quiet].sum() > 5 else float("nan"))
        print(f"  {name:20s} quiet-cell PR-AUC {q:.5f}   recently-burned {h:.5f}")

    # --- spatial CV: is the gain just location memorisation in another coat? ---
    print()
    print("=" * 74)
    print("SPATIAL CV (5 KMeans blocks, aggregated out-of-fold PR-AUC)")
    print("=" * 74)
    cells = df.drop_duplicates("grid_id")[["grid_id", "lat_center", "lon_center"]].copy()
    cells["fold"] = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(
        cells[["lat_center", "lon_center"]])
    df2 = df.merge(cells[["grid_id", "fold"]], on="grid_id")
    for name, cols in variants.items():
        oof = np.zeros(len(df2))
        for f in range(5):
            m = df2["fold"] == f
            oof[m.to_numpy()] = _fit_predict(df2[~m], df2[m], cols)
        print(f"  {name:20s} PR-AUC {average_precision_score(df2[TARGET_COL], oof):.5f}")


if __name__ == "__main__":
    main()
