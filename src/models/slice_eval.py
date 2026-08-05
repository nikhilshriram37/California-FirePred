"""Slice-based evaluation — the checks an aggregate metric cannot make.

Companion to :mod:`src.models.model_eval`, which guards against *memorisation*. This
module guards against the other failure that has actually shipped here: a model that
improves on every headline number while collapsing on the sub-population the product
exists for.

The 2026-08-05 recency retrain raised live PR-AUC by every aggregate measure and
simultaneously moved quiet-area ignitions from 17.5% Red to 0.3% Red. Nothing in the
headline could see it, and neither could a bootstrap on *ranking* — discrimination
within quiet cells barely moved (ROC 0.774 -> 0.759). What changed was where the
scores landed relative to the tier cutoffs. So:

  * :func:`quiet_mask` splits rows by whether anything actually burned nearby, using
    the causal recency panel — a property of the DATA, so the same split applies to
    every candidate model.
  * :func:`tier_counts` reports **actual tier assignment**, not just rank quality.
    This is the measure that would have caught the regression.
  * :func:`temporal_skill` is detrended, because fire counts fall steeply through
    autumn and a calendar-only model scores a large raw correlation for knowing the
    season rather than the weather.
  * :func:`spatial_skill` scores ranking *within* a day, isolating "where" from "when".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from src.models.recency import QUIET_EPS, is_quiet  # noqa: F401  (re-exported)

TIERS = ("red", "yellow", "green")


def quiet_mask(df: pd.DataFrame, eps: float = QUIET_EPS) -> np.ndarray:
    """Rows where neither the cell nor its neighbours have burned recently.

    Thin alias for :func:`src.models.recency.is_quiet` — the definition lives beside the
    features it reads so the serving path can import it without pulling in scipy.
    """
    return is_quiet(df, eps)


def calibrate_and_threshold(y: np.ndarray, raw: np.ndarray, *, red_recall: float,
                            yellow_recall: float, seed: int = 42):
    """Reproduce ``train.py``'s calibration and tier derivation on a holdout year.

    Isotonic fit on a stratified half, cutoffs read off that same half. Returns
    ``(calibrator, red_threshold, yellow_threshold)``.
    """
    cal_idx, _ = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y,
                                  random_state=seed)
    cal = IsotonicRegression(out_of_bounds="clip").fit(raw[cal_idx], y[cal_idx])
    p, yc = cal.transform(raw[cal_idx]), y[cal_idx]

    def thr(target: float) -> float:
        _, recall, t = precision_recall_curve(yc, p)
        ok = np.where(recall[:-1] >= target)[0]
        return float(t[ok[-1]]) if len(ok) else 0.0

    return cal, thr(red_recall), thr(yellow_recall)


def assign_tiers(risk: np.ndarray, red_t: float, yellow_t: float) -> np.ndarray:
    """Map calibrated risk to red / yellow / green."""
    return np.where(risk >= red_t, "red", np.where(risk >= yellow_t, "yellow", "green"))


def coverage_tiers(score: np.ndarray, red_cov: float, yellow_cov: float) -> np.ndarray:
    """Tiers defined by *coverage* — red = the top ``red_cov`` fraction of cell-days.

    Puts every candidate at the same operational cost, which is the only way to compare
    tier recall across models whose score distributions differ.
    """
    red_t = float(np.quantile(score, 1 - red_cov))
    yellow_t = float(np.quantile(score, 1 - red_cov - yellow_cov))
    return assign_tiers(score, red_t, yellow_t)


def hybrid_tiers(score: np.ndarray, quiet: np.ndarray, red_cov: float,
                 yellow_cov: float) -> np.ndarray:
    """Red stays a global absolute-risk statement; Yellow becomes regime-relative.

    Fully stratified tiering (:func:`coverage_tiers` applied within each regime) fixes
    quiet-area coverage but changes what Red *means*: measured on the live panel, a
    stratified Red cell in a quiet area carried a 0.86% observed fire rate against
    10.8% in an active one, so the same colour would denote a 12.6x difference in risk.

    Splitting the two cutoffs avoids that. Red keeps a single statewide meaning — the
    top ``red_cov`` of all cell-days — while Yellow is computed within regime, so a
    quiet area still surfaces its own most dangerous cells as "worth watching" rather
    than being uniformly Green. Overall red recall and lift are untouched; only the
    Yellow boundary moves.
    """
    red_t = float(np.quantile(score, 1 - red_cov))
    out = np.where(score >= red_t, "red", "green").astype(object)
    for m in (quiet, ~quiet):
        if not m.any():
            continue
        sub, cur = score[m], out[m]
        cut = float(np.quantile(sub, 1 - red_cov - yellow_cov))
        out[m] = np.where((cur != "red") & (sub >= cut), "yellow", cur)
    return out


def tier_counts(y: np.ndarray, tiers: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Where do the ignitions in this slice land? Counts, shares, and coverage."""
    m = np.ones(len(y), bool) if mask is None else mask
    yy, tt = y[m].astype(bool), tiers[m]
    n_fire = int(yy.sum())
    out = {"n_rows": int(m.sum()), "n_fires": n_fire}
    for t in TIERS:
        in_t = tt == t
        out[f"{t}_fires"] = int((in_t & yy).sum())
        out[f"{t}_share"] = float((in_t & yy).sum() / n_fire) if n_fire else float("nan")
        out[f"{t}_coverage"] = float(in_t.mean())
    # Lift = how much richer the red tier is than the slice's own base rate.
    base = yy.mean() if len(yy) else float("nan")
    red_rate = yy[tt == "red"].mean() if (tt == "red").any() else float("nan")
    out["red_lift"] = float(red_rate / base) if base else float("nan")
    return out


def rank_metrics(y: np.ndarray, score: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """PR-AUC / ROC-AUC on a slice, guarding against degenerate single-class slices."""
    m = np.ones(len(y), bool) if mask is None else mask
    yy, ss = y[m], score[m]
    if yy.sum() < 5 or yy.sum() == len(yy):
        return {"pr_auc": float("nan"), "roc_auc": float("nan"),
                "n_rows": int(m.sum()), "n_fires": int(yy.sum())}
    return {"pr_auc": float(average_precision_score(yy, ss)),
            "roc_auc": float(roc_auc_score(yy, ss)),
            "n_rows": int(m.sum()), "n_fires": int(yy.sum())}


def _detrend(v: np.ndarray, t: np.ndarray, deg: int = 2) -> np.ndarray:
    """Residuals after removing a low-order polynomial in time."""
    return v - np.polyval(np.polyfit(t, v, deg), t)


def temporal_skill(dates: np.ndarray, y: np.ndarray, score: np.ndarray,
                   deg: int = 2) -> dict:
    """Does the model know which DAYS are dangerous, beyond knowing the season?

    Correlates the predicted daily total against the actual daily ignition count, both
    raw and after removing a degree-``deg`` seasonal trend. The detrended figure is the
    honest one: over Sep-Nov the raw count falls by ~60%, so any model carrying a
    calendar scores a large raw correlation for nothing.
    """
    d = pd.DataFrame({"date": dates, "y": y, "s": score})
    g = d.groupby("date").agg(actual=("y", "sum"), pred=("s", "sum")).sort_index()
    t = np.arange(len(g), dtype=float)
    a, p = g["actual"].to_numpy(float), g["pred"].to_numpy(float)
    r_raw = stats.pearsonr(a, p)
    ra, rp = _detrend(a, t, deg), _detrend(p, t, deg)
    r_det = stats.pearsonr(ra, rp)
    return {"n_days": len(g), "r_raw": float(r_raw.statistic),
            "p_raw": float(r_raw.pvalue), "r_detrended": float(r_det.statistic),
            "p_detrended": float(r_det.pvalue)}


def spatial_skill(dates: np.ndarray, y: np.ndarray, score: np.ndarray) -> dict:
    """Ranking quality WITHIN a day, pooled over days — "where", with "when" removed.

    Per-day PR-AUC, averaged with each day weighted by its ignition count so a quiet
    day with one fire does not count as much as a busy one.
    """
    d = pd.DataFrame({"date": dates, "y": y, "s": score})
    num = den = 0.0
    for _, g in d.groupby("date"):
        yy = g["y"].to_numpy()
        if yy.sum() < 1 or yy.sum() == len(yy):
            continue
        num += average_precision_score(yy, g["s"].to_numpy()) * yy.sum()
        den += yy.sum()
    return {"within_day_pr_auc": float(num / den) if den else float("nan")}


def day_block_bootstrap(dates: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray,
                        *, n: int = 400, alpha: float = 0.05, seed: int = 42,
                        metric: str = "pr") -> dict:
    """CI for the metric gap between two score vectors, resampling whole DAYS.

    Days are the unit because thousands of cells scored on one day share a weather field
    and a fire cluster; a row-level bootstrap on this data shrinks the interval to
    nothing and calls noise significant.

    Array-based rather than the DataFrame version in :mod:`model_eval` so it stays
    usable on the 1.3M-row panels.
    """
    order = np.argsort(dates, kind="stable")
    ds = dates[order]
    y_s, a_s, b_s = y[order], a[order], b[order]
    _, starts = np.unique(ds, return_index=True)
    bounds = list(starts) + [len(ds)]
    idx = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(starts))]

    fn = average_precision_score if metric == "pr" else roc_auc_score
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        pick = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([idx[i] for i in pick])
        yy = y_s[sel]
        if yy.sum() < 5 or yy.sum() == len(yy):
            continue
        deltas.append(fn(yy, a_s[sel]) - fn(yy, b_s[sel]))
    if not deltas:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    dd = np.asarray(deltas)
    return {"median": float(np.median(dd)), "lo": float(np.quantile(dd, alpha / 2)),
            "hi": float(np.quantile(dd, 1 - alpha / 2)), "n": len(dd)}
