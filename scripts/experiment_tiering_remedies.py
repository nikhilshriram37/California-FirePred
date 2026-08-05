"""Two candidate fixes for the quiet-cell collapse, measured against the same slices.

The ablation says *which features to keep*. It cannot fix the symptom, because the
symptom is not a discrimination failure — the new model ranks quiet cells slightly
BETTER than the old one (PR 0.0047 -> 0.0064) and tiers them far worse. The scores are
fine; where they fall relative to a single global cutoff is not.

    A. Stratified tiering. Compute cutoffs *within* regime, so Red is the top slice of
       quiet cells plus the top slice of active cells rather than the top slice overall.
       Deployable as-is: the regime label comes from the recency features, which serving
       already computes. Total coverage is held fixed, so this is a pure reallocation —
       and it must cost overall recall, because active cells have ~10x the base rate.
       The point of measuring it is to price that trade rather than argue about it.

    B. Recency shape. ``days_since_fire_cell`` is the suspect: at its 365-day cap it
       actively pushes risk down, which is what turns "no recent fire" into Green. Tests
       the recency block with and without it, and with monotone constraints forcing
       recency to only ever add risk.

Run:  .venv/bin/python -m scripts.experiment_tiering_remedies
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data_acquisition.config import PROJECT_ROOT
from src.models.features import FEATURE_COLS, TARGET_COL
from src.models.slice_eval import QUIET_EPS, coverage_tiers, quiet_mask, rank_metrics, tier_counts
from src.models.train import TUNED_PARAMS
from scripts.analyze_feature_ablation import RED_COV, YELLOW_COV

logger = logging.getLogger(__name__)
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
OUT_DIR = EVAL_DIR / "ablation"
SEEDS = (42, 43, 44)


def stratified_tiers(score: np.ndarray, quiet: np.ndarray, red_cov: float,
                     yellow_cov: float, quiet_weight: float = 1.0) -> np.ndarray:
    """Tier within regime: the top ``red_cov`` of quiet cells AND of active cells.

    ``quiet_weight`` > 1 spends proportionally more of the red budget on quiet cells.
    At 1.0 the overall coverage is unchanged from global tiering — the same number of
    cells are flagged, just distributed differently.
    """
    out = np.empty(len(score), dtype=object)
    for m, w in ((quiet, quiet_weight), (~quiet, 1.0)):
        if not m.any():
            continue
        s = score[m]
        rc, yc = min(red_cov * w, 1.0), yellow_cov * w
        out[m] = coverage_tiers(s, rc, min(yc, 1.0 - rc))
    return out


def fit_seed_mean(tr, cols, panels, params=None):
    p = {**TUNED_PARAMS, **(params or {})}
    acc = [np.zeros(len(x)) for x in panels]
    for s in SEEDS:
        m = XGBClassifier(**p, tree_method="hist", eval_metric="aucpr", n_jobs=-1,
                          random_state=s)
        m.fit(tr[cols], tr[TARGET_COL])
        for i, x in enumerate(panels):
            acc[i] += m.predict_proba(x[cols])[:, 1] / len(SEEDS)
    return acc


def report(name, y, score, tiers, q, rows):
    tot = tier_counts(y, tiers)
    qc, ac = tier_counts(y, tiers, q), tier_counts(y, tiers, ~q)
    r = {"config": name, "red_coverage": tot["red_coverage"],
         "red_recall_all": tot["red_share"], "red_lift_all": tot["red_lift"],
         "red_share_quiet": qc["red_share"], "green_share_quiet": qc["green_share"],
         "red_share_active": ac["red_share"],
         "amber_share_quiet": 1 - qc["green_share"],
         "pr_quiet": rank_metrics(y, score, q)["pr_auc"],
         "pr_all": rank_metrics(y, score)["pr_auc"]}
    rows.append(r)
    print(f"{name:34s}{r['red_coverage']:>8.2%}{r['red_recall_all']:>9.1%}"
          f"{r['red_lift_all']:>8.1f}x{r['red_share_quiet']:>9.1%}"
          f"{r['amber_share_quiet']:>9.1%}{r['red_share_active']:>9.1%}{r['pr_quiet']:>9.4f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train = pd.read_parquet(EVAL_DIR / "panel_train201819.parquet")
    live = pd.read_parquet(EVAL_DIR / "panel_live2026.parquet")
    autumn = pd.read_parquet(EVAL_DIR / "panel_autumn2025.parquet")

    hdr = (f"{'configuration':34s}{'red cov':>8s}{'recall':>9s}{'lift':>9s}"
           f"{'Red q':>9s}{'R+Y q':>9s}{'Red act':>9s}{'PR q':>9s}")

    results = {}
    for pname, panel in (("live2026", live), ("autumn2025", autumn)):
        y, q = panel["confirmed"].to_numpy(), quiet_mask(panel, QUIET_EPS)
        rows: list[dict] = []

        print("\n" + "=" * 96)
        print(f"A. TIERING STRATEGY — {pname}, deployed 37-feature model, coverage held fixed")
        print("=" * 96)
        print(hdr)
        print("-" * 96)
        (score,) = fit_seed_mean(train, FEATURE_COLS, [panel])
        report("global cutoff (deployed)", y, score,
               coverage_tiers(score, RED_COV, YELLOW_COV), q, rows)
        for w in (1.0, 1.5, 2.0):
            report(f"stratified by regime (w={w})", y, score,
                   stratified_tiers(score, q, RED_COV, YELLOW_COV, w), q, rows)

        print("\n" + "=" * 96)
        print(f"B. RECENCY SHAPE — {pname}, global cutoff, so only the model changes")
        print("=" * 96)
        print(hdr)
        print("-" * 96)
        base = [c for c in FEATURE_COLS if c not in
                ("fire_recency_cell", "fire_recency_nbr", "days_since_fire_cell")]
        variants = {
            "no recency (34 feat)": (base, None),
            "recency, cell only": (base + ["fire_recency_cell"], None),
            "recency, cell+nbr (no days_since)": (base + ["fire_recency_cell", "fire_recency_nbr"], None),
            "full recency (deployed, 37)": (FEATURE_COLS, None),
        }
        # Monotone: recency may only ADD risk; days_since may only remove it as it grows.
        mono = {c: 0 for c in FEATURE_COLS}
        mono.update({"fire_recency_cell": 1, "fire_recency_nbr": 1, "days_since_fire_cell": -1})
        variants["full recency + monotone"] = (
            FEATURE_COLS, {"monotone_constraints": tuple(mono[c] for c in FEATURE_COLS)})

        for vname, (cols, extra) in variants.items():
            (s,) = fit_seed_mean(train, cols, [panel], extra)
            report(vname, y, s, coverage_tiers(s, RED_COV, YELLOW_COV), q, rows)

        # The combination the two halves suggest: best recency shape + stratified tiers.
        print("\n" + "=" * 96)
        print(f"C. COMBINED — {pname}")
        print("=" * 96)
        print(hdr)
        print("-" * 96)
        cols = base + ["fire_recency_cell", "fire_recency_nbr"]
        (s,) = fit_seed_mean(train, cols, [panel])
        report("cell+nbr recency, stratified", y, s,
               stratified_tiers(s, q, RED_COV, YELLOW_COV, 1.0), q, rows)
        (s2,) = fit_seed_mean(train, base, [panel])
        report("no recency, stratified (control)", y, s2,
               stratified_tiers(s2, q, RED_COV, YELLOW_COV, 1.0), q, rows)

        results[pname] = rows

    json.dump(results, open(OUT_DIR / "tiering_remedies.json", "w"), indent=2)
    logger.info("wrote tiering_remedies.json")


if __name__ == "__main__":
    main()
