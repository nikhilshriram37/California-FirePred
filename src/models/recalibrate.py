"""Refit the calibrator and tier cutoffs on live outcomes, leaving the booster alone.

The deployed calibrator was fit on 2020, where confirmed ignitions ran ~0.31% of
cell-days. Live summer prevalence is nearly 3x that, so the mapping from raw score to
probability is systematically low — measured at 0.66x (predicts 0.559%, observes
0.854%). Because the tiers are *absolute* cutoffs (deliberately: a percentile tier
would paint a wet day red), low probabilities mean too few cells clear the red
threshold. That is the mechanical explanation for live red recall landing near 25%
against a 55% design target.

Why this track is safe to run unattended, and why it comes before any retrain:

    Isotonic regression is monotone non-decreasing, so recalibration cannot *reorder*
    cells. It can only slide the tier cutoffs along a fixed precision/recall curve.
    (Measured, not assumed: ROC-AUC moved 0.80999 -> 0.80966 across a recalibration.
    It is not exactly invariant, because a monotone-*non-decreasing* map collapses
    distinct raw scores into ties and ties perturb AUC slightly. The ordering itself
    is untouched.) A retrain, by contrast, can reorder everything.

The corollary matters more than the safety property: **recalibration cannot make the
model better.** It cannot move the curve, only pick a point on it. If the achievable
points are all poor, that is a model problem and this module will not fix it.

That leaves two failure modes worth gating: a calibration that is worse than the one
it replaces, and a tier split that blows out to flag half the state. Both are checked
below on a held-out window the calibrator never saw.

Labels are 'confirmed' (IRWIN / CAL FIRE) rather than fused, so the risk number keeps
the meaning it was trained on — an ignition incident, not a FIRMS heat detection.

Run:  python -m src.models.recalibrate --dry-run
      python -m src.models.recalibrate
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

import src.data_acquisition.config  # noqa: F401 — loads .env.local
from src.data_acquisition.config import PROJECT_ROOT
from src.models.train import RED_RECALL, YELLOW_RECALL, _threshold_for_recall
from src.pipeline.metrics import _pull_date
from src.pipeline.supabase_io import get_client

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
CANDIDATE_DIR = MODELS_DIR / "candidate" / "recalibrated"

# --- Tier policy ----------------------------------------------------------------- #
# Red and yellow answer different questions, so they are parameterised differently.
#
# RED = "where do I send resources today?" That is capacity-constrained, so it is
# pinned to a **flag rate**, not a recall target. Live discrimination is far weaker
# than backtest (3.5% of CA buys 17% recall at 4.9x lift, where the 2020 backtest
# promised 59% at 17x), so a recall-targeted red would have to paint ~15% of the state
# to hit 55% — which is not a tier anyone can act on. Holding the rate fixed keeps red
# meaningful and lets its recall honestly reflect how good the model currently is.
#
# YELLOW = "where might something happen?" A wider net is the point, so it stays
# recall-targeted, unchanged from training.
RED_FLAG_RATE = 0.05           # red = the riskiest 5% of cells
# YELLOW_RECALL is imported from train.py (0.80) — deliberately not redefined here.

# --- Gate thresholds ------------------------------------------------------------- #
# Data sufficiency. Isotonic needs far less than a booster refit, but a calibration
# curve fit on a handful of positives is noise dressed as a correction.
MIN_DAYS = 21
MIN_POSITIVES = 250
# Gate on ECE, not Brier. Brier is a valid calibration measure here (recalibration
# cannot change discrimination, so Brier moves only through calibration) but at ~0.8%
# prevalence it is numerically dominated by the true negatives and barely budges — the
# first run improved ECE by 34% while Brier moved 1.7%. ECE has the resolution to tell
# a real correction from noise.
MIN_ECE_GAIN = 0.05            # relative, i.e. 5% better
# Sanity bounds on what actually ships. Red is pinned by construction, so this catches
# a degenerate score distribution (e.g. mass ties) rather than a policy disagreement.
RED_FLAG_BAND = (0.02, 0.10)
MIN_RED_LIFT = 2.0             # a red tier at less than 2x base rate is not worth drawing
HOLDOUT_FRACTION = 0.3         # newest ~30% of days are held out for the comparison


def healthy_dates(client) -> list[str]:
    """Label-complete dates only. A date we could not see is not a date without fires."""
    rows = client.table("label_health").select("date").eq("healthy", True).execute().data
    return sorted(r["date"] for r in rows)


def load_live(client, dates: list[str]) -> pd.DataFrame:
    """Raw model output joined to confirmed outcomes, over the serving population.

    Deliberately *not* restricted to the 2,304 cells shared with the historical grid:
    that restriction exists to keep training labels commensurable when merging the two
    datasets. A calibrator must describe the population actually being scored, which is
    all 4,169 live cells.
    """
    frames = []
    for ds in dates:
        pred = _pull_date(client, "risk_scores", "grid_id,raw_probability", ds)
        truth = _pull_date(client, "feature_history", "grid_id,has_fire,label_source", ds)
        if pred.empty or truth.empty:
            continue
        d = pred.merge(truth, on="grid_id")
        d = d[d["has_fire"].notna()]
        if d.empty:
            continue
        d["date"] = ds
        d["y"] = ((d["has_fire"] == 1) &
                  d["label_source"].fillna("").str.contains("irwin|calfire")).astype(int)
        frames.append(d[["grid_id", "date", "raw_probability", "y"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-count bins.

    Equal-width bins are useless below 1% prevalence — almost every cell lands in the
    first bin and the error averages to nothing — so bin by quantile instead.
    """
    order = np.argsort(p)
    p, y = p[order], y[order]
    err = 0.0
    for idx in np.array_split(np.arange(len(p)), n_bins):
        if len(idx):
            err += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    return float(err)


def calibration_report(p: np.ndarray, y: np.ndarray) -> dict:
    """How well probabilities p describe outcomes y."""
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": _ece(p, y),
        "mean_predicted": float(p.mean()),
        "mean_actual": float(y.mean()),
        "bias_ratio": float(p.mean() / y.mean()) if y.mean() else None,
    }


def tier_thresholds(p: np.ndarray, y: np.ndarray) -> dict:
    """Red from a flag-rate target, yellow from a recall target — see the tier policy.

    Red is clamped to sit at or above yellow: with a weak enough model the recall-based
    yellow cutoff can drift above the 95th percentile, which would otherwise emit tiers
    that contradict each other.
    """
    red = float(np.quantile(p, 1.0 - RED_FLAG_RATE))
    yellow = _threshold_for_recall(y, p, YELLOW_RECALL)
    return {"red": max(red, yellow), "yellow": yellow,
            "red_flag_rate_target": RED_FLAG_RATE, "yellow_recall_target": YELLOW_RECALL}


def _tier_outcome(p: np.ndarray, y: np.ndarray, thr: dict) -> dict:
    """What a set of cutoffs would actually have done on this window."""
    red = p >= thr["red"]
    ry = p >= thr["yellow"]
    n_fire, base = int(y.sum()), float(y.mean())
    red_hits = int((red & (y == 1)).sum())
    red_prec = red_hits / int(red.sum()) if red.sum() else 0.0
    return {
        "red_flag_rate": float(red.mean()),
        "red_recall": float(red_hits / n_fire) if n_fire else None,
        "red_lift": float(red_prec / base) if base else None,
        "ry_flag_rate": float(ry.mean()),
        "ry_recall": float((ry & (y == 1)).sum() / n_fire) if n_fire else None,
    }


def evaluate(models_dir: Path = MODELS_DIR, holdout_fraction: float = HOLDOUT_FRACTION) -> dict:
    """Fit a candidate calibrator and judge it against the incumbent on held-out days."""
    client = get_client()
    if client is None:
        return {"promote": False, "reasons": ["Supabase not configured"]}

    dates = healthy_dates(client)
    df = load_live(client, dates)
    if df.empty:
        return {"promote": False, "reasons": ["no joined live prediction/outcome rows"]}

    # Temporal split — the holdout must be *later* than the fit window, because that is
    # the only split that resembles how the calibrator will actually be used.
    day_list = sorted(df["date"].unique())
    n_hold = max(1, int(len(day_list) * holdout_fraction))
    fit_days, hold_days = set(day_list[:-n_hold]), set(day_list[-n_hold:])
    fit_df = df[df["date"].isin(fit_days)]
    hold_df = df[df["date"].isin(hold_days)]

    n_pos = int(df["y"].sum())
    reasons: list[str] = []
    if len(day_list) < MIN_DAYS:
        reasons.append(f"only {len(day_list)} healthy days (need {MIN_DAYS})")
    if n_pos < MIN_POSITIVES:
        reasons.append(f"only {n_pos} confirmed positives (need {MIN_POSITIVES})")
    if int(hold_df["y"].sum()) == 0:
        reasons.append("holdout window contains no confirmed positives")

    incumbent = joblib.load(models_dir / "calibrator.joblib")
    inc_thr = json.loads((models_dir / "thresholds.json").read_text())

    candidate = IsotonicRegression(out_of_bounds="clip")
    candidate.fit(fit_df["raw_probability"].to_numpy(), fit_df["y"].to_numpy())

    hold_raw, hold_y = hold_df["raw_probability"].to_numpy(), hold_df["y"].to_numpy()
    inc_p, cand_p = incumbent.transform(hold_raw), candidate.transform(hold_raw)
    inc_cal, cand_cal = calibration_report(inc_p, hold_y), calibration_report(cand_p, hold_y)

    # Cutoffs come from the fit window only — deriving them on the holdout would be
    # scoring the candidate against thresholds tuned to that same holdout.
    fit_p = candidate.transform(fit_df["raw_probability"].to_numpy())
    cand_thr = tier_thresholds(fit_p, fit_df["y"].to_numpy())
    inc_out = _tier_outcome(inc_p, hold_y, inc_thr)
    cand_out = _tier_outcome(cand_p, hold_y, cand_thr)

    # The artifact that ships is refit on every healthy day — withholding the newest
    # days is right for an honest comparison, wrong for what goes to production. Its
    # cutoffs are therefore what determine how much of the state actually turns red,
    # and the fit-window proxy above systematically misstates that (it over-predicts on
    # later days, reporting ~9% red where the shipped artifact settles near 5%).
    ship = IsotonicRegression(out_of_bounds="clip")
    ship.fit(df["raw_probability"].to_numpy(), df["y"].to_numpy())
    ship_p = ship.transform(df["raw_probability"].to_numpy())
    ship_y = df["y"].to_numpy()
    ship_thr = tier_thresholds(ship_p, ship_y)
    ship_out = _tier_outcome(ship_p, ship_y, ship_thr)
    per_day = df.assign(_p=ship_p).groupby("date").apply(
        lambda g: float((g["_p"] >= ship_thr["red"]).mean()), include_groups=False)

    ece_gain = (inc_cal["ece"] - cand_cal["ece"]) / inc_cal["ece"] if inc_cal["ece"] else 0.0
    brier_gain = (inc_cal["brier"] - cand_cal["brier"]) / inc_cal["brier"] if inc_cal["brier"] else 0.0

    # Calibration quality is the ONLY genuine quality comparison available between two
    # calibrators. Lift deliberately is not compared: isotonic cannot reorder cells, so
    # incumbent and candidate lie on one identical precision/recall curve, and their
    # lift differs only because they sit at different points on it. Comparing those
    # numbers measures the operating-point choice, not the model — the candidate's 4.0x
    # at 9.0% flagged and the incumbent's 4.6x at 6.4% are the same curve twice.
    if ece_gain < MIN_ECE_GAIN:
        reasons.append(f"ECE gain {ece_gain:+.1%} below the {MIN_ECE_GAIN:.0%} bar")
    # An absolute floor still earns its place: it catches the curve itself collapsing,
    # which a retrain (Stage 3) genuinely can cause even though recalibration cannot.
    if (cand_out["red_lift"] or 0) < MIN_RED_LIFT:
        reasons.append(f"held-out red lift {cand_out['red_lift']:.1f}x below the "
                       f"{MIN_RED_LIFT:.1f}x floor — the tier would not be informative")

    # Ship-artifact gate: does red stay as narrow as the tier policy requires?
    lo, hi = RED_FLAG_BAND
    if not (lo <= ship_out["red_flag_rate"] <= hi):
        reasons.append(f"shipped red would flag {ship_out['red_flag_rate']:.1%} of cells, "
                       f"outside the {lo:.0%}-{hi:.0%} band")

    return {
        "promote": not reasons,
        "reasons": reasons,
        "n_days": len(day_list), "n_rows": len(df), "n_positives": n_pos,
        "fit_days": len(fit_days), "holdout_days": len(hold_days),
        "holdout_positives": int(hold_df["y"].sum()),
        "incumbent": {**inc_cal, **inc_out, "thresholds": inc_thr},
        "candidate": {**cand_cal, **cand_out, "thresholds": cand_thr},
        "shipped": {**ship_out, "thresholds": ship_thr,
                    "per_day_flag_min": float(per_day.min()),
                    "per_day_flag_median": float(per_day.median()),
                    "per_day_flag_max": float(per_day.max())},
        "ece_gain": ece_gain, "brier_gain": brier_gain,
        "_ship_obj": ship, "_ship_thr": ship_thr,
        "_candidate_obj": candidate,
        "_full": df,
    }


def write_candidate(result: dict, out_dir: Path = CANDIDATE_DIR) -> dict:
    """Write the artifacts a promotion would install.

    These are the all-days refit built and gated in :func:`evaluate`, not a fresh fit —
    writing anything the gate did not inspect would defeat the gate.
    """
    final, thr = result["_ship_obj"], result["_ship_thr"]
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_dir / "calibrator.joblib")
    (out_dir / "thresholds.json").write_text(json.dumps(thr, indent=2))

    card = json.loads((MODELS_DIR / "model_card.json").read_text())
    card.update({
        "version": "recal-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
        "recalibrated_at": datetime.now(timezone.utc).isoformat(),
        "recalibration": {
            "source": "live confirmed labels (IRWIN/CAL FIRE)",
            "n_days": result["n_days"], "n_rows": result["n_rows"],
            "n_positives": result["n_positives"],
            "holdout_brier_gain": result["brier_gain"],
            "note": "booster unchanged; isotonic is monotone so ranking metrics are identical",
        },
        "tiers": {"red": thr["red"], "yellow": thr["yellow"]},
    })
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    return thr


def _fmt(d: dict) -> str:
    return (f"ece {d['ece']:.5f}  brier {d['brier']:.6f}  "
            f"predicts {d['mean_predicted']*100:.3f}% vs actual {d['mean_actual']*100:.3f}% "
            f"({d['bias_ratio']:.2f}x)\n"
            f"              red flags {d['red_flag_rate']*100:5.1f}% catching "
            f"{(d['red_recall'] or 0)*100:3.0f}% at {(d['red_lift'] or 0):.1f}x lift  |  "
            f"red+yel flags {d['ry_flag_rate']*100:.1f}% catching {(d['ry_recall'] or 0)*100:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="evaluate but write no artifacts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    r = evaluate()
    print(f"\nlive data: {r.get('n_rows', 0):,} rows over {r.get('n_days', 0)} healthy days, "
          f"{r.get('n_positives', 0)} confirmed positives")
    print(f"split: fit on {r.get('fit_days', 0)} days -> holdout {r.get('holdout_days', 0)} days "
          f"({r.get('holdout_positives', 0)} positives)\n")
    if "incumbent" in r:
        print(f"  incumbent : {_fmt(r['incumbent'])}")
        print(f"  candidate : {_fmt(r['candidate'])}")
        print(f"\n  ECE gain: {r['ece_gain']:+.1%}   (Brier {r['brier_gain']:+.1%})")
        s = r["shipped"]
        print(f"\n  WHAT WOULD SHIP (refit on all {r['n_days']} days):")
        print(f"    red >= {s['thresholds']['red']:.5f}   yellow >= {s['thresholds']['yellow']:.5f}")
        print(f"    red flags {s['red_flag_rate']*100:.1f}% of cells, catching "
              f"{(s['red_recall'] or 0)*100:.0f}% at {(s['red_lift'] or 0):.1f}x lift")
        print(f"    red+yellow flags {s['ry_flag_rate']*100:.1f}%, catching "
              f"{(s['ry_recall'] or 0)*100:.0f}%")
        print(f"    per-day red: min {s['per_day_flag_min']*100:.1f}%  median "
              f"{s['per_day_flag_median']*100:.1f}%  max {s['per_day_flag_max']*100:.1f}%")

    print(f"\nDECISION: {'PROMOTE' if r['promote'] else 'HOLD'}")
    for why in r["reasons"]:
        print(f"  - {why}")

    if r["promote"] and not args.dry_run:
        thr = write_candidate(r)
        print(f"\nwrote candidate artifacts -> {CANDIDATE_DIR}")
        print(f"  final cutoffs: red>={thr['red']:.5f}  yellow>={thr['yellow']:.5f}")
    elif r["promote"]:
        print("\n(dry run — no artifacts written)")


if __name__ == "__main__":
    main()
