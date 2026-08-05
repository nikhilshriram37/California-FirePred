"""Honest-evaluation primitives for judging one model against another.

Extracted from the retrain track (removed 2026-08-03) because these two checks are
the part worth keeping: they are what tell you a candidate model is genuinely better
rather than better-looking. Use them whenever a new model is proposed — e.g. the
fire-recency retrain — before believing a headline metric.

Both exist because a headline PR-AUC comparison on fire data is untrustworthy:

  * fires cluster in time. Thousands of cells scored on one day share a single
    weather field, so rows are nowhere near independent and any row-level
    significance test manufactures confidence out of that correlation.
  * fires cluster in space, and recur. A cell that burned recently is likely to burn
    again (median same-cell repeat gap: 15 days; 35% within a week). A model can post
    a large gain purely by learning *which cells were active*, which is memorisation,
    not prediction — and it decays away the moment the season turns.

Recorded so the lesson is not re-learned the hard way: a candidate once posted a
+146% live PR-AUC gain that survived a bootstrap and a full-year backtest, and was
almost entirely memorisation:

    cells that burned during training   0.054 -> 0.127   (+133%)
    cells that did not                  0.0071 -> 0.0077 (+9%, CI straddling zero)

The full-year backtest could not see it (it predates the window, so contains none of
the memorised cells) and neither could the headline number (recurrence put 63% of the
holdout's fires on already-seen cells). Only :func:`seen_unseen_split` caught it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.models.features import TARGET_COL

BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.05   # one-sided 95% lower bound on the PR-AUC delta


def block_bootstrap_delta(df: pd.DataFrame, *, cand: str = "p_cand", inc: str = "p_inc",
                          y: str = TARGET_COL, date: str = "date",
                          n: int = BOOTSTRAP_N, alpha: float = BOOTSTRAP_ALPHA,
                          seed: int = 42) -> tuple[float, float, float]:
    """Distribution of the candidate-minus-incumbent PR-AUC gap, resampling whole days.

    Days are the resampling unit on purpose — see the module docstring. A row-level
    bootstrap on the same data will shrink the interval to nothing and call noise
    significant.

    Args:
        df: rows carrying the two score columns, the label, and a date column.
        cand / inc: column names of the candidate and incumbent scores.
        y: binary label column. date: the blocking unit.
        n: bootstrap resamples. alpha: one-sided lower-bound quantile.

    Returns:
        (median delta, lower bound, upper bound). Treat the candidate as a real
        improvement only when the lower bound clears zero.
    """
    rng = np.random.default_rng(seed)
    days = df[date].unique()
    by_day = {d: df[df[date] == d] for d in days}
    deltas = []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        s = pd.concat([by_day[d] for d in pick], ignore_index=True)
        if s[y].nunique() < 2:
            continue
        yy = s[y].to_numpy()
        deltas.append(average_precision_score(yy, s[cand].to_numpy()) -
                      average_precision_score(yy, s[inc].to_numpy()))
    if not deltas:
        return 0.0, 0.0, 0.0
    d = np.array(deltas)
    return (float(np.median(d)), float(np.quantile(d, alpha)),
            float(np.quantile(d, 1 - alpha)))


def seen_unseen_split(train: pd.DataFrame, hold: pd.DataFrame, *, cand: str = "p_cand",
                      inc: str = "p_inc", y: str = TARGET_COL, cell: str = "grid_id",
                      date: str = "date") -> dict:
    """Split a holdout by whether each cell burned during training, and score each half.

    The ``unseen`` half is the one that matters: it asks whether the candidate is
    better on cells it never watched burn. A candidate whose gain lives entirely in
    the ``seen`` half has memorised locations rather than learned to predict them.

    The unseen half also gets its own :func:`block_bootstrap_delta`, because a
    *concentration* failure is not a *regression* — an earlier version of this check
    tested only for regression on unseen cells and therefore passed the exact
    candidate it was written to reject. Gate on ``unseen["delta_lo"] > 0``.

    Note the split is only meaningful when training and holdout are separated by an
    embargo gap of at least a few days; otherwise a single multi-day fire lands on
    both sides and pollutes the "seen" set with its own continuation.
    """
    seen_cells = set(train.loc[train[y] == 1, cell])
    hold = hold.assign(_seen=hold[cell].isin(seen_cells))
    out: dict = {"n_seen_cells": len(seen_cells)}
    for key, sub in [("seen", hold[hold["_seen"]]), ("unseen", hold[~hold["_seen"]])]:
        yy = sub[y].to_numpy()
        if yy.sum() < 5:
            out[key] = None
            continue
        a = float(average_precision_score(yy, sub[inc]))
        b = float(average_precision_score(yy, sub[cand]))
        entry = {"n_rows": len(sub), "n_fires": int(yy.sum()),
                 "incumbent_pr_auc": a, "candidate_pr_auc": b,
                 "change": (b - a) / a if a else None}
        if key == "unseen":
            med, lo, hi = block_bootstrap_delta(sub, cand=cand, inc=inc, y=y, date=date)
            entry.update({"delta_median": med, "delta_lo": lo, "delta_hi": hi})
        out[key] = entry
    return out
