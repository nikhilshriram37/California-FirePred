"""Read the ablation's raw scores and answer the questions an aggregate metric cannot.

Reporting principles, each of which exists because ignoring it has already cost this
project a shipped regression:

  * Nothing is reported pooled-only. Every combination is scored on quiet cells and
    active cells separately, because the deployed model improved on every pooled number
    while collapsing on quiet ones.
  * Tier assignment is reported, not just PR-AUC. Ranking within quiet cells barely
    moved during the regression (ROC 0.774 -> 0.759); what moved was where the scores
    landed relative to the cutoffs.
  * Tiers are compared at MATCHED COVERAGE. Models whose score distributions differ
    flag different shares of the state at their own cutoffs, so their raw tier tables
    compare operational cost as much as skill.
  * Blocks are judged by their marginal contribution averaged over the combinations
    that differ only in that block — not by which single combination tops a column.
    With 31 combinations x several metrics x several slices, the top of any one column
    is mostly noise.
  * Seed spread is carried through, so a difference smaller than XGBoost's own
    randomness can be dismissed rather than explained.

Run:  .venv/bin/python -m scripts.analyze_feature_ablation
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.data_acquisition.config import PROJECT_ROOT
from src.models.slice_eval import (QUIET_EPS, assign_tiers, coverage_tiers,
                                   day_block_bootstrap, quiet_mask, rank_metrics,
                                   spatial_skill, temporal_skill, tier_counts)
from scripts.run_feature_ablation import BLOCKS, ORDER, combos

logger = logging.getLogger(__name__)
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "ablation"

# The operating point production actually paid over the live window: red on 5.80% of
# cell-days, yellow on the next 22.45%. Every model is re-tiered to this so the tier
# comparison is like-for-like.
RED_COV, YELLOW_COV = 0.0580, 0.2245

PANELS = ("live2026", "autumn2025", "holdout2020", "live2026_cold")
FULL, NO_REC = "WGCLR", "WGCL"


def load_panel(name: str, quiet_from: pd.DataFrame | None = None):
    df = pd.read_parquet(OUT_DIR / f"scores_{name}.parquet")
    if quiet_from is None:
        q = quiet_mask(df, QUIET_EPS)
    else:
        # live2026_cold carries the starved prior in its own columns; slice it on the
        # truthful (warm) one so the two are compared on the same population.
        key = df[["grid_id", "date"]].merge(
            quiet_from.assign(_q=quiet_mask(quiet_from, QUIET_EPS))[["grid_id", "date", "_q"]],
            on=["grid_id", "date"], how="left")
        q = key["_q"].to_numpy(dtype=bool)
    return df, q


def metrics_for(df: pd.DataFrame, q: np.ndarray, col: str, cal, red_t: float,
                yel_t: float) -> dict:
    y = df["confirmed"].to_numpy()
    raw = df[col].to_numpy()
    dates = df["date"].to_numpy()
    deg = 2 if df["date"].nunique() <= 120 else 6

    a, qm, am = rank_metrics(y, raw), rank_metrics(y, raw, q), rank_metrics(y, raw, ~q)
    m = {"pr_all": a["pr_auc"], "roc_all": a["roc_auc"],
         "pr_quiet": qm["pr_auc"], "roc_quiet": qm["roc_auc"],
         "pr_active": am["pr_auc"], "roc_active": am["roc_auc"],
         "within_day_pr": spatial_skill(dates, y, raw)["within_day_pr_auc"]}
    ts = temporal_skill(dates, y, raw, deg=deg)
    m["r_temporal_raw"] = ts["r_raw"]
    m["r_temporal_det"] = ts["r_detrended"]
    m["p_temporal_det"] = ts["p_detrended"]

    # Tiers at matched coverage — the like-for-like comparison.
    tt = coverage_tiers(raw, RED_COV, YELLOW_COV)
    for slab, mask in (("quiet", q), ("active", ~q)):
        tc = tier_counts(y, tt, mask)
        m[f"red_share_{slab}"] = tc["red_share"]
        m[f"green_share_{slab}"] = tc["green_share"]
        m[f"n_fires_{slab}"] = tc["n_fires"]

    # Tiers at the model's OWN cutoffs — reproduces the deployed procedure, and shows
    # the calibration behaviour that the matched-coverage view deliberately removes.
    risk = cal.transform(raw)
    to = assign_tiers(risk, red_t, yel_t)
    m["own_red_coverage"] = float((to == "red").mean())
    for slab, mask in (("quiet", q), ("active", ~q)):
        m[f"own_red_share_{slab}"] = tier_counts(y, to, mask)["red_share"]
    return m


def block_effects(tbl: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Marginal contribution of each block, averaged over matched combination pairs.

    For block b, every combination containing b is paired with the combination that is
    identical except for b — 15 pairs (the block-alone combination has no
    non-empty complement). Averaging those deltas is a far more stable read
    than the top of a leaderboard, and it is the only way to say "recency is worth X"
    rather than "this one combination happened to win".
    """
    s = tbl.set_index("combo")[metric]
    rows = []
    for b in ORDER:
        deltas = []
        for c in combos():
            if b in c:
                base = c.replace(b, "")
                if base and base in s.index and np.isfinite(s[c]) and np.isfinite(s[base]):
                    deltas.append(s[c] - s[base])
        if deltas:
            d = np.array(deltas)
            rows.append({"block": b, "n_pairs": len(d), "mean_delta": d.mean(),
                         "median_delta": np.median(d), "min": d.min(), "max": d.max(),
                         "frac_positive": float((d > 0).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    thr = json.loads((OUT_DIR / "thresholds.json").read_text())
    names = combos()

    warm_ref, _ = load_panel("live2026")
    tables: dict[str, pd.DataFrame] = {}
    for panel in PANELS:
        df, q = load_panel(panel, quiet_from=warm_ref if panel == "live2026_cold" else None)
        logger.info("%s: %s rows, %s ignitions, quiet %.1f%% of rows holding %s ignitions",
                    panel, f"{len(df):,}", int(df.confirmed.sum()), 100 * q.mean(),
                    int(df.confirmed.to_numpy()[q].sum()))
        rows = []
        for n in names:
            cal = joblib.load(OUT_DIR / f"calibrator_{n}.joblib")
            m = metrics_for(df, q, f"s_{n}", cal, thr[n]["red"], thr[n]["yellow"])
            rows.append({"combo": n, "n_features": thr[n]["n_features"], **m})
        tables[panel] = pd.DataFrame(rows)
        tables[panel].to_csv(OUT_DIR / f"metrics_{panel}.csv", index=False)

    # Seed spread, so differences smaller than the fit's own randomness get dismissed.
    ps = pd.read_parquet(OUT_DIR / "per_seed_pr.parquet")
    spread = (ps.groupby(["panel", "combo"])["pr_auc"].std().groupby("panel").median())
    print("\n" + "=" * 96)
    print("SEED NOISE — median across combos of the per-combo PR-AUC sd over 3 seeds")
    print("=" * 96)
    for k, v in spread.items():
        print(f"  {k:16s} {v:.5f}")

    for panel in PANELS:
        t = tables[panel]
        print("\n" + "=" * 96)
        print(f"{panel.upper()}  —  every combination, quiet vs active")
        print("=" * 96)
        print(f"{'combo':7s}{'nf':>4s}{'PR all':>9s}{'PR quiet':>10s}{'PR act':>9s}"
              f"{'ROCq':>7s}{'inday':>8s}{'r_det':>7s}{'Red%q':>7s}{'Red%a':>7s}{'Grn%q':>7s}")
        print("-" * 96)
        for _, r in t.sort_values("pr_quiet", ascending=False).iterrows():
            print(f"{r.combo:7s}{int(r.n_features):>4d}{r.pr_all:>9.4f}{r.pr_quiet:>10.4f}"
                  f"{r.pr_active:>9.4f}{r.roc_quiet:>7.3f}{r.within_day_pr:>8.4f}"
                  f"{r.r_temporal_det:>7.2f}{r.red_share_quiet:>7.1%}"
                  f"{r.red_share_active:>7.1%}{r.green_share_quiet:>7.1%}")

        print(f"\nMARGINAL BLOCK CONTRIBUTION ({panel}) — mean over 16 matched pairs")
        for metric in ("pr_all", "pr_quiet", "pr_active", "within_day_pr",
                       "r_temporal_det", "red_share_quiet", "red_share_active"):
            be = block_effects(t, metric)
            cells = "  ".join(f"{r.block}:{r.mean_delta:+.4f}({r.frac_positive:.0%})"
                              for _, r in be.iterrows())
            print(f"  {metric:16s} {cells}")

    json.dump({p: tables[p].to_dict("records") for p in PANELS},
              open(OUT_DIR / "metrics_all.json", "w"), indent=2)
    logger.info("wrote metrics_*.csv and metrics_all.json")


if __name__ == "__main__":
    main()
