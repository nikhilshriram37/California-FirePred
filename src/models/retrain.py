"""Retrain the booster on historical + live data, and judge it against the incumbent.

This is the only track that can actually improve the model. Recalibration slides the
tier cutoffs along a fixed precision/recall curve; retraining is what moves the curve
itself — and the curve badly needs moving. Live discrimination sits at PR-AUC ~0.031
against a 0.091 backtest, and normalised for prevalence that is 3.6x better than
random live versus 29x in backtest.

Merging live data into training is only sound under two corrections, both measured
rather than assumed:

  * confirmed labels only. Fused labels (IRWIN+CALFIRE+FIRMS) run 2.65x the historical
    base rate, because FIRMS sees industrial and agricultural heat no weather model can
    predict. Confirmed labels (IRWIN/CAL FIRE) run 1.11x — commensurable.
  * the 2,304 cells shared with the historical grid. The live grid has 4,169 cells and
    the historical set 3,668; only 2,304 are common. The live-only cells have no
    historical counterpart, a different base rate, and incomplete coverage on the four
    days that predate the grid expansion.

Promotion requires clearing a dual holdout:

  1. A recent live window the candidate never trained on, where it must beat the
     incumbent by a margin that survives a **day-level block bootstrap**. Cells within
     a day share weather, so they are nowhere near independent; a row-level bootstrap
     would manufacture significance out of that correlation.
  2. The 2020 full-year backtest, as a seasonal guard — live data is summer-only, and a
     model that learns summer at winter's expense must not pass. This is a *bounded*
     regression, not a strict one: live and 2020 distributions differ enough that a
     genuinely better production model can legitimately score slightly worse on 2020,
     and a strict bar would veto exactly the improvement we are looking for.

Run:  python -m src.models.retrain --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

import src.data_acquisition.config  # noqa: F401 — loads .env.local
from src.data_acquisition.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.features import FEATURE_COLS, TARGET_COL, merge_static_features
from src.models.train import TRAIN_YEARS, TEST_YEAR, TUNED_PARAMS
from src.pipeline.supabase_io import get_client

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
CANDIDATE_DIR = MODELS_DIR / "candidate" / "retrained"

# --- Gate thresholds ------------------------------------------------------------- #
MIN_LIVE_DAYS = 30
MIN_LIVE_POSITIVES = 400        # confirmed, on the shared footprint
LIVE_HOLDOUT_DAYS = 12          # newest live days, withheld from training
BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.05          # one-sided 95% lower bound on the PR-AUC delta
MAX_BACKTEST_REGRESSION = 0.05  # relative; see the seasonal-guard note above
# A candidate must not go backwards on cells that did *not* burn during live training.
# See :func:`spatial_generalization` for why this gate exists and what it caught.
MAX_UNSEEN_REGRESSION = 0.05
_PAGE = 1000


def load_live(client) -> pd.DataFrame:
    """Live feature rows with confirmed labels, on the shared footprint, healthy days.

    ``feature_history.features`` is a jsonb blob written by the scoring pipeline from
    the same FEATURE_COLS contract the model consumes, so it expands straight into
    training columns with no re-derivation.
    """
    healthy = sorted(r["date"] for r in
                     client.table("label_health").select("date").eq("healthy", True).execute().data)
    hist_cells = set(pd.read_parquet(PROCESSED_DIR / "california_dataset.parquet",
                                     columns=["grid_id"]).grid_id.unique())
    frames = []
    for ds in healthy:
        rows, frm = [], 0
        while True:
            page = (client.table("feature_history")
                    .select("grid_id,features,has_fire,label_source")
                    .eq("date", ds).order("grid_id").range(frm, frm + _PAGE - 1).execute().data)
            rows += page
            if len(page) < _PAGE:
                break
            frm += _PAGE
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d = d[d["grid_id"].isin(hist_cells) & d["has_fire"].notna()]
        if d.empty:
            continue
        feats = pd.DataFrame(list(d["features"]), index=d.index).reindex(columns=FEATURE_COLS)
        feats["grid_id"] = d["grid_id"].to_numpy()
        feats["date"] = ds
        feats[TARGET_COL] = ((d["has_fire"] == 1) &
                             d["label_source"].fillna("").str.contains("irwin|calfire")).astype(int).to_numpy()
        frames.append(feats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_historical() -> pd.DataFrame:
    """The original training frame, static features merged, unchanged."""
    df = pd.read_parquet(PROCESSED_DIR / "california_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return merge_static_features(df)


def _fit(X: pd.DataFrame, y: np.ndarray) -> XGBClassifier:
    m = XGBClassifier(**TUNED_PARAMS, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42)
    m.fit(X, y)
    return m


def block_bootstrap_delta(hold: pd.DataFrame, n: int = BOOTSTRAP_N,
                          seed: int = 42) -> tuple[float, float, float]:
    """Distribution of the candidate-minus-incumbent PR-AUC gap, resampling whole days.

    Days are the resampling unit on purpose. Thousands of cells scored on the same day
    share one weather field, so treating rows as independent would shrink the interval
    to nothing and declare noise significant.

    Returns (median delta, lower bound, upper bound).
    """
    rng = np.random.default_rng(seed)
    days = hold["date"].unique()
    by_day = {d: hold[hold["date"] == d] for d in days}
    deltas = []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        s = pd.concat([by_day[d] for d in pick], ignore_index=True)
        if s[TARGET_COL].nunique() < 2:
            continue
        y = s[TARGET_COL].to_numpy()
        deltas.append(average_precision_score(y, s["p_cand"].to_numpy()) -
                      average_precision_score(y, s["p_inc"].to_numpy()))
    if not deltas:
        return 0.0, 0.0, 0.0
    d = np.array(deltas)
    return (float(np.median(d)),
            float(np.quantile(d, BOOTSTRAP_ALPHA)),
            float(np.quantile(d, 1 - BOOTSTRAP_ALPHA)))


def spatial_generalization(live_train: pd.DataFrame, hold: pd.DataFrame) -> dict:
    """Split the live holdout by whether a cell burned during live training.

    This gate exists because the first candidate it was run against posted a +146%
    live PR-AUC gain of which almost none was real prediction:

        cells that burned in live training   0.054 -> 0.127   (+133%)
        cells that did not                   0.0071 -> 0.0077 (+9%)

    With no "recent fire in this cell" feature available, the booster encodes that
    signal through lat/lon — the spatial reliance the project's spatial CV already
    documented — and memorises which cells were active in the live window. That is
    not a model that predicts fire; it is a model that remembers where fire was, and
    it will keep those cells hot after the season turns.

    Crucially the 2020 seasonal guard **cannot** see this: 2020 predates the live
    window entirely, so it contains none of the memorised cells. Without this check an
    autonomous gate would promote such a candidate believing the headline gain.
    """
    seen_cells = set(live_train.loc[live_train[TARGET_COL] == 1, "grid_id"])
    hold = hold.assign(_seen=hold["grid_id"].isin(seen_cells))
    out: dict = {"n_seen_cells": len(seen_cells)}
    for key, sub in [("seen", hold[hold["_seen"]]), ("unseen", hold[~hold["_seen"]])]:
        y = sub[TARGET_COL].to_numpy()
        if y.sum() < 5:
            out[key] = None
            continue
        a = float(average_precision_score(y, sub["p_inc"]))
        b = float(average_precision_score(y, sub["p_cand"]))
        out[key] = {"n_rows": len(sub), "n_fires": int(y.sum()),
                    "incumbent_pr_auc": a, "candidate_pr_auc": b,
                    "change": (b - a) / a if a else None}
    return out


def evaluate() -> dict:
    """Train a candidate on historical + live, and judge it on the dual holdout."""
    client = get_client()
    if client is None:
        return {"promote": False, "reasons": ["Supabase not configured"]}

    hist = load_historical()
    live = load_live(client)
    reasons: list[str] = []
    if live.empty:
        return {"promote": False, "reasons": ["no usable live training rows"]}

    live_days = sorted(live["date"].unique())
    n_live_pos = int(live[TARGET_COL].sum())
    if len(live_days) < MIN_LIVE_DAYS:
        reasons.append(f"only {len(live_days)} healthy live days (need {MIN_LIVE_DAYS})")
    if n_live_pos < MIN_LIVE_POSITIVES:
        reasons.append(f"only {n_live_pos} confirmed live positives (need {MIN_LIVE_POSITIVES})")

    hold_days = set(live_days[-LIVE_HOLDOUT_DAYS:])
    live_train = live[~live["date"].isin(hold_days)]
    live_hold = live[live["date"].isin(hold_days)].copy()

    hist_train = hist[hist["date"].dt.year.isin(TRAIN_YEARS)]
    backtest = hist[hist["date"].dt.year == TEST_YEAR]

    # Candidate sees the original training years plus the live window minus its holdout.
    X_tr = pd.concat([hist_train[FEATURE_COLS], live_train[FEATURE_COLS]], ignore_index=True)
    y_tr = np.concatenate([hist_train[TARGET_COL].to_numpy(), live_train[TARGET_COL].to_numpy()])
    logger.info("training candidate on %s historical + %s live rows (%s + %s positives)",
                f"{len(hist_train):,}", f"{len(live_train):,}",
                f"{int(hist_train[TARGET_COL].sum()):,}", f"{int(live_train[TARGET_COL].sum()):,}")
    candidate = _fit(X_tr, y_tr)

    incumbent = XGBClassifier()
    incumbent.load_model(str(MODELS_DIR / "xgb_model.json"))

    # Holdout 1 — recent live days, never trained on by either model.
    live_hold["p_inc"] = incumbent.predict_proba(live_hold[FEATURE_COLS])[:, 1]
    live_hold["p_cand"] = candidate.predict_proba(live_hold[FEATURE_COLS])[:, 1]
    y_h = live_hold[TARGET_COL].to_numpy()
    live_res = {
        "n_rows": len(live_hold), "n_fires": int(y_h.sum()), "n_days": len(hold_days),
        "incumbent_pr_auc": float(average_precision_score(y_h, live_hold["p_inc"])),
        "candidate_pr_auc": float(average_precision_score(y_h, live_hold["p_cand"])),
        "incumbent_roc_auc": float(roc_auc_score(y_h, live_hold["p_inc"])),
        "candidate_roc_auc": float(roc_auc_score(y_h, live_hold["p_cand"])),
    }
    med, lo, hi = block_bootstrap_delta(live_hold)
    live_res.update({"delta_median": med, "delta_lo": lo, "delta_hi": hi})

    # Holdout 2 — 2020 full year, the seasonal guard.
    Xb, yb = backtest[FEATURE_COLS], backtest[TARGET_COL].to_numpy()
    bt = {
        "incumbent_pr_auc": float(average_precision_score(yb, incumbent.predict_proba(Xb)[:, 1])),
        "candidate_pr_auc": float(average_precision_score(yb, candidate.predict_proba(Xb)[:, 1])),
    }
    bt["regression"] = ((bt["incumbent_pr_auc"] - bt["candidate_pr_auc"])
                        / bt["incumbent_pr_auc"]) if bt["incumbent_pr_auc"] else 0.0

    # Holdout 3 — does the gain survive on cells the live window never saw burn?
    spatial = spatial_generalization(live_train, live_hold)

    if lo <= 0:
        reasons.append(f"live PR-AUC gain {med:+.4f} is not significant "
                       f"(95% lower bound {lo:+.4f} does not clear zero)")
    if bt["regression"] > MAX_BACKTEST_REGRESSION:
        reasons.append(f"2020 backtest regresses {bt['regression']:.1%}, past the "
                       f"{MAX_BACKTEST_REGRESSION:.0%} seasonal-guard allowance")
    if spatial.get("unseen") is None:
        reasons.append("too few holdout fires on unseen cells to test generalisation")
    elif -(spatial["unseen"]["change"] or 0) > MAX_UNSEEN_REGRESSION:
        reasons.append(f"regresses {-spatial['unseen']['change']:.1%} on cells that did not "
                       f"burn during live training — the gain is memorisation, not prediction")

    return {
        "promote": not reasons, "reasons": reasons,
        "live_days": len(live_days), "live_rows": len(live), "live_positives": n_live_pos,
        "live_holdout": live_res, "backtest": bt, "spatial": spatial,
        "_candidate": candidate, "_hist": hist, "_live": live,
    }


def write_candidate(result: dict, out_dir: Path = CANDIDATE_DIR) -> dict:
    """Write the artifacts a promotion would install, calibrated on the live window."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model, live = result["_candidate"], result["_live"]
    model.save_model(str(out_dir / "xgb_model.json"))

    raw = model.predict_proba(live[FEATURE_COLS])[:, 1]
    y = live[TARGET_COL].to_numpy()
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw, y)
    joblib.dump(cal, out_dir / "calibrator.joblib")

    from src.models.recalibrate import tier_thresholds
    thr = tier_thresholds(cal.transform(raw), y)
    (out_dir / "thresholds.json").write_text(json.dumps(thr, indent=2))
    (out_dir / "feature_list.json").write_text(json.dumps(FEATURE_COLS, indent=2))
    (out_dir / "model_card.json").write_text(json.dumps({
        "model": "xgboost-isotonic",
        "version": "retrain-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_years": TRAIN_YEARS,
        "live_days": result["live_days"], "live_positives": result["live_positives"],
        "n_features": len(FEATURE_COLS), "params": TUNED_PARAMS,
        "metrics": {"live_holdout": result["live_holdout"],
                    "backtest_2020": result["backtest"],
                    "generalisation": result.get("spatial")},
        "known_limitation": (
            "The live gain is concentrated in cells that burned during the live training "
            "window. With no recent-fire feature available the booster encodes that through "
            "lat/lon, so those cells stay elevated instead of decaying once conditions "
            "change. Adding an explicit decaying recency feature is the principled fix; "
            "until then this model should be re-trained often and watched by rollback."),
        "tiers": {"red": thr["red"], "yellow": thr["yellow"]},
    }, indent=2))
    return thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="evaluate but write no artifacts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    r = evaluate()
    if "live_holdout" in r:
        lh, bt = r["live_holdout"], r["backtest"]
        print(f"\nlive data: {r['live_rows']:,} rows over {r['live_days']} days, "
              f"{r['live_positives']} confirmed positives (shared footprint)")
        print(f"\nHOLDOUT 1 — live, {lh['n_days']} days / {lh['n_rows']:,} rows / {lh['n_fires']} fires")
        print(f"  PR-AUC   incumbent {lh['incumbent_pr_auc']:.5f}  ->  candidate {lh['candidate_pr_auc']:.5f}")
        print(f"  ROC-AUC  incumbent {lh['incumbent_roc_auc']:.5f}  ->  candidate {lh['candidate_roc_auc']:.5f}")
        print(f"  delta    {lh['delta_median']:+.5f}  (day-block bootstrap 95% CI "
              f"[{lh['delta_lo']:+.5f}, {lh['delta_hi']:+.5f}])")
        print(f"\nHOLDOUT 2 — 2020 backtest (seasonal guard)")
        print(f"  PR-AUC   incumbent {bt['incumbent_pr_auc']:.5f}  ->  candidate {bt['candidate_pr_auc']:.5f}"
              f"   ({bt['regression']:+.1%} regression)")

        sp = r.get("spatial", {})
        print(f"\nHOLDOUT 3 — generalisation (memorisation check), "
              f"{sp.get('n_seen_cells', 0)} cells burned during live training")
        for key, label in [("seen", "cells that burned in training"),
                           ("unseen", "cells that did not          ")]:
            d = sp.get(key)
            print(f"  {label}  " + ("too few fires to test" if not d else
                  f"{d['n_fires']:4d} fires  {d['incumbent_pr_auc']:.5f} -> "
                  f"{d['candidate_pr_auc']:.5f}  ({d['change']:+.0%})"))

    print(f"\nDECISION: {'PROMOTE' if r['promote'] else 'HOLD'}")
    for why in r["reasons"]:
        print(f"  - {why}")

    if r["promote"] and not args.dry_run:
        thr = write_candidate(r)
        print(f"\nwrote candidate artifacts -> {CANDIDATE_DIR}")
        print(f"  cutoffs: red>={thr['red']:.5f}  yellow>={thr['yellow']:.5f}")


if __name__ == "__main__":
    main()
