"""Fit every combination of the model's feature blocks and score them on three regimes.

Diagnostic, not a tuning run. The question is not "which combination wins" — a single
leaderboard number is what let the quiet-cell tier collapse ship in the first place —
but *which block carries which kind of skill, in which regime*. The scoring here is
deliberately dumb: it writes raw per-row predictions to disk and leaves every judgement
to :mod:`scripts.analyze_feature_ablation`.

Blocks (5 -> 31 non-empty combinations):

    W  weather (22)     the gridMET fire-weather feed and its rolling derivatives
    G  geography (8)    lat/lon + topography + log population — static per cell
    C  calendar (3)     month, month_sin, doy_cos
    L  lightning (1)    lightning_count
    R  recency (3)      the current fire prior added 2026-08-05

Held fixed across every combination: the tuned hyperparameters, the 2018-19 fitting
window, the calibration procedure, and the tier-derivation procedure. Only the feature
set varies. Note the hyperparameters were tuned for the full 37-feature set, which
mildly favours it — worth stating, not worth re-tuning per combo, since a per-combo
search would confound feature effects with search luck.

Three seeds per combination, because with 31 combinations several metrics and several
slices, XGBoost's own randomness will manufacture apparent winners. Seed spread is
reported so a difference smaller than it can be dismissed.

Run:  .venv/bin/python scripts/run_feature_ablation.py
Out:  data/eval/ablation/  (raw scores per panel, calibrators, thresholds)
"""

from __future__ import annotations

import itertools
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data_acquisition.config import PROJECT_ROOT
from src.models.features import STATIC_FEATURES, TARGET_COL
from src.models.recency import RECENCY_FEATURES, merge_recency
from src.models.slice_eval import calibrate_and_threshold
from src.models.train import RED_RECALL, TUNED_PARAMS, YELLOW_RECALL

logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
OUT_DIR = EVAL_DIR / "ablation"

SEEDS = (42, 43, 44)

BLOCKS: dict[str, list[str]] = {
    "W": ["rmin", "vs", "pr", "vpd", "fm100", "bi", "aet", "water_deficit", "tmmx_c",
          "erc_7d", "erc_14d", "vpd_7d", "vpd_14d", "bi_7d", "bi_14d", "tmmx_7d",
          "rmin_7d", "dry_streak", "pr_7d", "pr_14d", "fm100_change_3d", "vpd_change_3d"],
    "G": ["lat_center", "lon_center", *STATIC_FEATURES],
    "C": ["month", "month_sin", "doy_cos"],
    "L": ["lightning_count"],
    "R": RECENCY_FEATURES,
}
ORDER = "WGCLR"


def combos() -> list[str]:
    """All 31 non-empty block combinations, named by their block letters."""
    out = []
    for k in range(1, len(ORDER) + 1):
        for c in itertools.combinations(ORDER, k):
            out.append("".join(c))
    return out


def cols_for(name: str) -> list[str]:
    return [c for b in ORDER if b in name for c in BLOCKS[b]]


def load_panels() -> dict[str, pd.DataFrame]:
    """The fitting panel plus the four evaluation panels.

    ``live2026_cold`` re-derives the fire-recency prior from in-window labels only,
    reproducing what production actually had at launch (``feature_history`` begins
    2026-06-19). Comparing it against ``live2026`` separates "the recency block is
    wrong" from "the recency block was starved".
    """
    p = {n: pd.read_parquet(EVAL_DIR / f"panel_{n}.parquet")
         for n in ("train201819", "holdout2020", "live2026", "autumn2025")}
    warm = p["live2026"]
    cold = warm.drop(columns=RECENCY_FEATURES)
    p["live2026_cold"] = merge_recency(
        cold, warm.loc[warm["confirmed"] == 1, ["grid_id", "date"]], warmup_days=150)
    for n, d in p.items():
        logger.info("%-15s %9s rows  %5s ignitions", n, f"{len(d):,}", int(d["confirmed"].sum()))
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panels = load_panels()
    train = panels.pop("train201819")
    y_train = train[TARGET_COL].to_numpy()
    y2020 = panels["holdout2020"][TARGET_COL].to_numpy()

    names = combos()
    scores = {n: {} for n in panels}          # panel -> combo -> seed-mean raw score
    per_seed: list[dict] = []
    thresholds: dict[str, dict] = {}

    t_start = time.time()
    for i, name in enumerate(names, 1):
        cols = cols_for(name)
        t0 = time.time()
        acc = {n: np.zeros(len(d)) for n, d in panels.items()}
        for s in SEEDS:
            m = XGBClassifier(**TUNED_PARAMS, tree_method="hist", eval_metric="aucpr",
                              n_jobs=-1, random_state=s)
            m.fit(train[cols], y_train)
            for n, d in panels.items():
                p = m.predict_proba(d[cols])[:, 1]
                acc[n] += p / len(SEEDS)
                per_seed.append({"combo": name, "seed": s, "panel": n,
                                 "pr_auc": _safe_pr(d["confirmed"].to_numpy(), p)})
        for n in panels:
            scores[n][name] = acc[n].astype(np.float32)

        # Calibrate and derive tier cutoffs on the 2020 holdout, exactly as train.py.
        cal, red_t, yel_t = calibrate_and_threshold(
            y2020, acc["holdout2020"], red_recall=RED_RECALL, yellow_recall=YELLOW_RECALL)
        joblib.dump(cal, OUT_DIR / f"calibrator_{name}.joblib")
        thresholds[name] = {"red": red_t, "yellow": yel_t, "n_features": len(cols)}

        logger.info("[%2d/%2d] %-6s %2d feats  %.0fs  (elapsed %.1f min)",
                    i, len(names), name, len(cols), time.time() - t0,
                    (time.time() - t_start) / 60)

    for n, d in panels.items():
        keep = ["grid_id", "date", "confirmed", "fire_recency_cell", "fire_recency_nbr",
                "days_since_fire_cell"]
        out = d[keep].copy()
        for name, v in scores[n].items():
            out[f"s_{name}"] = v
        out.to_parquet(OUT_DIR / f"scores_{n}.parquet", index=False)
        logger.info("wrote scores_%s.parquet", n)

    (OUT_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    pd.DataFrame(per_seed).to_parquet(OUT_DIR / "per_seed_pr.parquet", index=False)
    logger.info("done in %.1f min", (time.time() - t_start) / 60)


def _safe_pr(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


if __name__ == "__main__":
    main()
