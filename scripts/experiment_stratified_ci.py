"""Confidence intervals on the stratified-tiering recommendation.

The remedy experiment reports point estimates: at identical coverage, computing tier
cutoffs within regime moves quiet-area ignitions from 0.5% Red to 25.4% Red and costs
about 8 points of overall red recall. Those are the numbers a decision would be made
on, so they need intervals — and the intervals have to resample whole DAYS, because
fires cluster in time and a row-level bootstrap would call almost anything significant.

Reuses the ablation's saved scores rather than refitting, so this measures the tiering
rule and nothing else.

Run:  .venv/bin/python -m scripts.experiment_stratified_ci
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.data_acquisition.config import PROJECT_ROOT
from src.models.slice_eval import QUIET_EPS, coverage_tiers, hybrid_tiers, quiet_mask
from scripts.analyze_feature_ablation import RED_COV, YELLOW_COV
from scripts.experiment_tiering_remedies import stratified_tiers

logger = logging.getLogger(__name__)
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "ablation"
N_BOOT = 500


def stats_for(y, tiers, q):
    red, amber = tiers == "red", tiers != "green"
    nq, na = y[q].sum(), y.sum()
    return {
        "red_recall_all": float((red & (y == 1)).sum() / na) if na else np.nan,
        "red_share_quiet": float((red & (y == 1) & q).sum() / nq) if nq else np.nan,
        "amber_share_quiet": float((amber & (y == 1) & q).sum() / nq) if nq else np.nan,
        "red_coverage": float(red.mean()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = {}
    for panel in ("live2026", "autumn2025"):
        df = pd.read_parquet(OUT_DIR / f"scores_{panel}.parquet")
        q = quiet_mask(df, QUIET_EPS)
        y = df["confirmed"].to_numpy()
        s_new = df["s_WGCLR"].to_numpy()
        s_old = df["s_WGCL"].to_numpy()

        arms = {
            "old model, global": (s_old, None),
            "new model, global (deployed)": (s_new, None),
            "new model, stratified": (s_new, "strat"),
            "new model, HYBRID": (s_new, "hybrid"),
        }
        # Cutoffs are derived on the FULL panel once, then held fixed across resamples —
        # a bootstrap that re-derived them each time would measure threshold jitter
        # rather than the difference between the two rules.
        rule = {"strat": lambda s: stratified_tiers(s, q, RED_COV, YELLOW_COV),
                "hybrid": lambda s: hybrid_tiers(s, q, RED_COV, YELLOW_COV),
                None: lambda s: coverage_tiers(s, RED_COV, YELLOW_COV)}
        tiers = {k: rule[how](s) for k, (s, how) in arms.items()}

        point = {k: stats_for(y, t, q) for k, t in tiers.items()}

        days = df["date"].to_numpy()
        idx = {d: np.where(days == d)[0] for d in np.unique(days)}
        keys = list(idx)
        rng = np.random.default_rng(42)
        boot: dict[str, list] = {k: [] for k in arms}
        for _ in range(N_BOOT):
            sel = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            if y[sel].sum() < 5:
                continue
            for k, t in tiers.items():
                boot[k].append(stats_for(y[sel], t[sel], q[sel]))

        print("\n" + "=" * 94)
        print(f"{panel.upper()} — tiering rules at identical coverage, 95% CI over {N_BOOT} day-resamples")
        print("=" * 94)
        print(f"{'configuration':32s}{'red recall':>22s}{'Red % of quiet ign':>24s}{'R+Y % of quiet ign':>24s}")
        print("-" * 94)
        res = {}
        for k in arms:
            b = pd.DataFrame(boot[k])
            row = {}
            cells = []
            for m in ("red_recall_all", "red_share_quiet", "amber_share_quiet"):
                lo, hi = np.nanquantile(b[m], [0.025, 0.975])
                row[m] = {"point": point[k][m], "lo": float(lo), "hi": float(hi)}
                cells.append(f"{point[k][m]:6.1%} [{lo:5.1%},{hi:5.1%}]")
            res[k] = row
            print(f"{k:32s}{cells[0]:>22s}{cells[1]:>24s}{cells[2]:>24s}")

        # Paired deltas: same resample, both rules, so the day-to-day variation cancels.
        print(f"\n  paired deltas (HYBRID minus deployed, same resamples):")
        for m, lbl in (("red_recall_all", "overall red recall"),
                       ("red_share_quiet", "Red share of quiet ignitions"),
                       ("amber_share_quiet", "R+Y share of quiet ignitions")):
            d = np.array([a[m] - c[m] for a, c in
                          zip(boot["new model, HYBRID"], boot["new model, global (deployed)"])])
            lo, hi = np.nanquantile(d, [0.025, 0.975])
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "tie"
            print(f"    {lbl:32s}{np.nanmedian(d):+7.1%}  [{lo:+6.1%}, {hi:+6.1%}]  {sig}")
        out[panel] = res

    json.dump(out, open(OUT_DIR / "stratified_ci.json", "w"), indent=2)
    logger.info("wrote stratified_ci.json")


if __name__ == "__main__":
    main()
