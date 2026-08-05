"""Is the quiet-cell tier collapse the model, or a starved fire-recency prior?

The 2026-08-05 retrain moved quiet-area ignitions from 17.5% Red to 0.3% Red. That was
measured against the recency prior as *served*, and serving reads trailing confirmed
fires from ``feature_history`` — a table that only starts 2026-06-19. So at launch the
model was told ``days_since_fire_cell = 365`` (the cap) for 88.7% of California, while
4,298 real CA ignitions had occurred since January. That feature pushes risk down.

Two very different diagnoses fit the same symptom:

    (a) the model over-weights recency and suppresses everything without a recent fire
        -> needs stratified tiering or monotonic constraints (restructuring)
    (b) the prior was starved at deployment and self-heals as history accumulates
        -> needs a longer fire history at score time (a data fix, far cheaper)

This decomposes the observed change into three independent steps:

    served            what production actually published (old model, corrupted features)
    old  + corrected  isolates the serving-feature corruption
    new  + COLD       the new model with the prior it actually had at launch
    new  + WARM       the same model with a properly warmed prior (Jan-Jun history)

Run:  .venv/bin/python scripts/experiment_recency_coldstart.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data_acquisition.config import PROJECT_ROOT
from src.models.features import FEATURE_COLS, TARGET_COL
from src.models.recency import RECENCY_FEATURES, merge_recency
from src.models.slice_eval import (QUIET_EPS, assign_tiers, calibrate_and_threshold,
                                   coverage_tiers,
                                   quiet_mask, rank_metrics, tier_counts)
from src.models.train import RED_RECALL, TRAIN_YEARS, TUNED_PARAMS, YELLOW_RECALL

logger = logging.getLogger(__name__)
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

OLD_COLS = [c for c in FEATURE_COLS if c not in RECENCY_FEATURES]   # the pre-recency 34
SEEDS = (42, 43, 44)


def fit_seed_mean(tr: pd.DataFrame, cols: list[str], panels: list[pd.DataFrame]):
    """Fit over several seeds and return the mean raw probability for each panel."""
    acc = [np.zeros(len(p)) for p in panels]
    for s in SEEDS:
        m = XGBClassifier(**TUNED_PARAMS, tree_method="hist", eval_metric="aucpr",
                          n_jobs=-1, random_state=s)
        m.fit(tr[cols], tr[TARGET_COL])
        for i, p in enumerate(panels):
            acc[i] += m.predict_proba(p[cols])[:, 1] / len(SEEDS)
    return acc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train = pd.read_parquet(EVAL_DIR / "panel_train201819.parquet")
    h2020 = pd.read_parquet(EVAL_DIR / "panel_holdout2020.parquet")
    warm = pd.read_parquet(EVAL_DIR / "panel_live2026.parquet")

    # The prior as production actually had it: in-window labels only.
    cold = warm.drop(columns=RECENCY_FEATURES)
    cold = merge_recency(cold, warm.loc[warm.confirmed == 1, ["grid_id", "date"]],
                         warmup_days=150)
    logger.info("cold prior: %.1f%% of rows at the days_since cap vs %.1f%% warm",
                100 * (cold.days_since_fire_cell >= 365).mean(),
                100 * (warm.days_since_fire_cell >= 365).mean())

    y = warm["confirmed"].to_numpy()
    y2020 = h2020[TARGET_COL].to_numpy()
    # Slice on the WARM prior: "did anything really burn nearby" is a fact about the
    # world, not about what the model was fed. The cold-prior split is reported too,
    # because that is the definition the original regression was recorded against.
    q_warm, q_cold = quiet_mask(warm, QUIET_EPS), quiet_mask(cold, QUIET_EPS)

    runs: dict[str, np.ndarray] = {}

    logger.info("fitting old (34 feature) model")
    r2020_old, rlive_old = fit_seed_mean(train, OLD_COLS, [h2020, warm])
    cal_o, red_o, yel_o = calibrate_and_threshold(
        y2020, r2020_old, red_recall=RED_RECALL, yellow_recall=YELLOW_RECALL)
    runs["old + corrected"] = cal_o.transform(rlive_old)

    logger.info("fitting new (37 feature) model")
    r2020_new, rlive_cold, rlive_warm = fit_seed_mean(train, FEATURE_COLS, [h2020, cold, warm])
    cal_n, red_n, yel_n = calibrate_and_threshold(
        y2020, r2020_new, red_recall=RED_RECALL, yellow_recall=YELLOW_RECALL)
    runs["new + COLD prior"] = cal_n.transform(rlive_cold)
    runs["new + WARM prior"] = cal_n.transform(rlive_warm)

    thresholds = {"old + corrected": (red_o, yel_o),
                  "new + COLD prior": (red_n, yel_n),
                  "new + WARM prior": (red_n, yel_n)}

    # What production actually published over this window (old model, corrupted features).
    served = pd.read_parquet(EVAL_DIR / "live_record.parquet")
    served = served.sort_values(["date", "grid_id"]).reset_index(drop=True)
    key = warm.sort_values(["date", "grid_id"]).reset_index(drop=True)
    assert (served.grid_id.to_numpy() == key.grid_id.to_numpy()).all(), "panel/served row mismatch"
    served_tiers = served["tier"].str.lower().to_numpy()
    served_risk = served["risk"].to_numpy()
    y_served = key["confirmed"].to_numpy()
    qs_warm = quiet_mask(key, QUIET_EPS)

    rows = []
    print("\n" + "=" * 100)
    print(f"QUIET-CELL TIER ASSIGNMENT  (quiet = no fire in cell or neighbours, eps={QUIET_EPS})")
    print("=" * 100)
    print(f"{'configuration':22s} {'slice':7s} {'ign':>5s} {'Red':>12s} {'Yellow':>12s} "
          f"{'Green':>12s} {'red cov':>8s} {'PR-AUC':>8s}")
    print("-" * 100)

    def emit(name, tiers, risk, yy, qm):
        for slab, m in (("quiet", qm), ("active", ~qm)):
            tc = tier_counts(yy, tiers, m)
            rm = rank_metrics(yy, risk, m)
            print(f"{name:22s} {slab:7s} {tc['n_fires']:5d} "
                  f"{tc['red_fires']:5d} ({tc['red_share']:5.1%}) "
                  f"{tc['yellow_fires']:5d} ({tc['yellow_share']:5.1%}) "
                  f"{tc['green_fires']:5d} ({tc['green_share']:5.1%}) "
                  f"{tc['red_coverage']:7.2%} {rm['pr_auc']:8.4f}")
            rows.append({"config": name, "slice": slab, **tc, **rm})

    emit("served (as published)", served_tiers, served_risk, y_served, qs_warm)
    for name, risk in runs.items():
        red_t, yel_t = thresholds[name]
        emit(name, assign_tiers(risk, red_t, yel_t), risk, y, q_warm)

    print("\n" + "=" * 100)
    print("SAME, sliced by the COLD prior — the definition the original regression used")
    print("=" * 100)
    print(f"{'configuration':22s} {'slice':7s} {'ign':>5s} {'Red':>12s} {'Yellow':>12s} {'Green':>12s}")
    print("-" * 100)
    for name, risk in runs.items():
        red_t, yel_t = thresholds[name]
        tiers = assign_tiers(risk, red_t, yel_t)
        for slab, m in (("quiet", q_cold), ("active", ~q_cold)):
            tc = tier_counts(y, tiers, m)
            print(f"{name:22s} {slab:7s} {tc['n_fires']:5d} "
                  f"{tc['red_fires']:5d} ({tc['red_share']:5.1%}) "
                  f"{tc['yellow_fires']:5d} ({tc['yellow_share']:5.1%}) "
                  f"{tc['green_fires']:5d} ({tc['green_share']:5.1%})")

    # Each configuration flags a different share of the state red, so the tables above
    # partly compare operational cost rather than skill. Re-tier everything at the
    # coverage production actually paid, and the comparison becomes like-for-like.
    red_cov = float((served_tiers == "red").mean())
    yel_cov = float((served_tiers == "yellow").mean())
    print("\n" + "=" * 100)
    print(f"MATCHED COVERAGE — every model re-tiered to red={red_cov:.2%} / yellow={yel_cov:.2%} "
          "of cell-days (what production actually paid)")
    print("=" * 100)
    print(f"{'configuration':22s} {'slice':7s} {'ign':>5s} {'Red':>12s} {'Yellow':>12s} {'Green':>12s}")
    print("-" * 100)
    for name, risk in [("served (as published)", served_risk), *runs.items()]:
        yy, qm = (y_served, qs_warm) if name.startswith("served") else (y, q_warm)
        tiers = coverage_tiers(risk, red_cov, yel_cov)
        for slab, m in (("quiet", qm), ("active", ~qm)):
            tc = tier_counts(yy, tiers, m)
            print(f"{name:22s} {slab:7s} {tc['n_fires']:5d} "
                  f"{tc['red_fires']:5d} ({tc['red_share']:5.1%}) "
                  f"{tc['yellow_fires']:5d} ({tc['yellow_share']:5.1%}) "
                  f"{tc['green_fires']:5d} ({tc['green_share']:5.1%})")

    out = PROJECT_ROOT / "data" / "eval" / "coldstart_decomposition.json"
    out.write_text(json.dumps(rows, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
