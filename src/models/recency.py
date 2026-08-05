"""Fire-recency features: a *current* spatial prior, decaying over time.

Why this exists. Measured on the live record (2026-08-04), a model built from static
per-cell features alone — topography, population, lat/lon, no weather at all — beat the
deployed weather model by 5.2x on PR-AUC. On the live target, *where* dominates *when*:
ignitions recur in the same places, and the model's spatial prior was frozen at whatever
2018-2020 looked like. It had no way to know that a cell burned last week, so it smuggled
location in through lat/lon, where nothing ever decays and a cell that burned in 2019
stays hot forever.

These features carry that signal explicitly and let it fade:

    fire_recency_cell     decayed count of recent ignitions in the cell
    fire_recency_nbr      the same over the 8 surrounding cells (fires cross boundaries)
    days_since_fire_cell  time since the cell last burned, capped

Causality. Every value for day t is built strictly from days at or before ``t - lag_days``
via ``r[t] = r[t-1] * decay + M[t - lag_days]``. Nothing from day t enters, so the feature
is servable — and ``lag_days`` is deliberately explicit rather than assumed to be 1:
scoring runs at 13:00/21:00 UTC and labelling at 15:00/23:00, so the freshest label at
scoring time is a day or two behind. Training must apply the *same* lag, or the model
learns from a recency signal fresher than the one it will ever be served — which is the
train/serve skew class of bug this feature was introduced to fix.

Forecasting falls out for free: pass target dates beyond the last observed fire and the
recursion simply decays forward, which is the correct belief about an unknown future.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Decay constant, in days. Selected on the live holdout across three temporal splits:
# 7 -> 60 all beat the incumbent, with PR-AUC rising gently over that range. 60 edged
# 30 by +0.010 PR (95% CI [+0.0004, +0.0216]) and by nothing at all on ROC, which is
# too thin a margin to justify the longer memory: 30 keeps the feature a genuine
# *recency* signal rather than a season-cumulative prior, and depends far less on a
# long warm-up being available.
TAU_DAYS = 30.0

# Cells that have never burned get this rather than NaN — a real "long time ago"
# rather than a missing value XGBoost would route down its default path.
DAYS_SINCE_CAP = 365.0

# Labels trail scoring in production; see the module docstring.
DEFAULT_LAG_DAYS = 2

RECENCY_FEATURES: list[str] = [
    "fire_recency_cell", "fire_recency_nbr", "days_since_fire_cell",
]

CELL_DEG = 0.1


def _neighbour_index(cells: np.ndarray, centers: pd.DataFrame) -> list[list[int]]:
    """For each cell, the column positions of its 8 grid neighbours."""
    ix = np.round(centers["lon_center"].to_numpy() / CELL_DEG).astype(int)
    iy = np.round(centers["lat_center"].to_numpy() / CELL_DEG).astype(int)
    pos = {(x, y): j for j, (x, y) in enumerate(zip(ix, iy))}
    return [[pos[(x + dx, y + dy)]
             for dx in (-1, 0, 1) for dy in (-1, 0, 1)
             if (dx or dy) and (x + dx, y + dy) in pos]
            for x, y in zip(ix, iy)]


def recency_panel(
    fires: pd.DataFrame,
    centers: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    tau: float = TAU_DAYS,
    lag_days: int = DEFAULT_LAG_DAYS,
) -> pd.DataFrame:
    """Recency features for every (cell, date) in ``dates``.

    One implementation serves training, live scoring and forecasting, so the three can
    never drift apart.

    Args:
        fires: observed ignitions, one row per (grid_id, date) that burned. Dates
            outside ``dates`` are still used if they precede it — that history is
            exactly what the decay needs.
        centers: grid_id, lat_center, lon_center for every cell to emit.
        dates: the full contiguous daily index to compute over. Must start early
            enough to warm the decay up (a few multiples of ``tau``), and may extend
            past the last observed fire, in which case the prior decays forward.
        tau: decay constant in days.
        lag_days: how stale the freshest usable label is assumed to be.

    Returns:
        Frame with grid_id, date and :data:`RECENCY_FEATURES`.
    """
    centers = centers.drop_duplicates("grid_id").sort_values("grid_id").reset_index(drop=True)
    cells = centers["grid_id"].to_numpy()
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    n_d, n_c = len(dates), len(cells)

    # Dense (day x cell) ignition matrix. Fires outside the grid or the window are
    # dropped rather than silently wrapped.
    M = np.zeros((n_d, n_c), dtype=np.float32)
    if len(fires):
        f = fires.copy()
        f["date"] = pd.to_datetime(f["date"])
        di = pd.Index(dates).get_indexer(f["date"])
        ci = pd.Index(cells).get_indexer(f["grid_id"])
        ok = (di >= 0) & (ci >= 0)
        np.add.at(M, (di[ok], ci[ok]), 1.0)
        if (~ok).any():
            logger.debug("recency: ignored %d fire row(s) outside the grid/date window",
                         int((~ok).sum()))

    nbr_of = _neighbour_index(cells, centers)
    N = np.zeros_like(M)
    for j, nb in enumerate(nbr_of):
        if nb:
            N[:, j] = M[:, nb].sum(axis=1)

    decay = float(np.exp(-1.0 / tau))
    R = np.zeros_like(M)
    RN = np.zeros_like(M)
    DS = np.full((n_d, n_c), DAYS_SINCE_CAP, dtype=np.float32)
    for t in range(1, n_d):
        src = t - lag_days                      # the freshest day we are allowed to use
        add = M[src] if src >= 0 else 0.0
        add_n = N[src] if src >= 0 else 0.0
        R[t] = R[t - 1] * decay + add
        RN[t] = RN[t - 1] * decay + add_n
        burned = (M[src] > 0) if src >= 0 else np.zeros(n_c, dtype=bool)
        DS[t] = np.where(burned, float(lag_days),
                         np.minimum(DS[t - 1] + 1.0, DAYS_SINCE_CAP))

    return pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), n_c),
        "grid_id": np.tile(cells, n_d),
        "fire_recency_cell": R.ravel(),
        "fire_recency_nbr": RN.ravel(),
        "days_since_fire_cell": DS.ravel(),
    })


def merge_recency(
    df: pd.DataFrame,
    fires: pd.DataFrame,
    *,
    tau: float = TAU_DAYS,
    lag_days: int = DEFAULT_LAG_DAYS,
    warmup_days: int = 120,
) -> pd.DataFrame:
    """Attach recency features to a frame carrying grid_id, date, lat/lon centers.

    ``warmup_days`` of history before the frame's first date are included in the
    recursion so the earliest rows are not artificially cold. At tau=14 a fire decays
    to under 0.03% of its initial weight in 120 days, so nothing meaningful is lost.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    start = df["date"].min() - pd.Timedelta(days=warmup_days)
    dates = pd.date_range(start, df["date"].max(), freq="D")
    panel = recency_panel(fires, df[["grid_id", "lat_center", "lon_center"]],
                          dates, tau=tau, lag_days=lag_days)
    out = df.merge(panel, on=["grid_id", "date"], how="left")
    for c in RECENCY_FEATURES:
        fill = DAYS_SINCE_CAP if c == "days_since_fire_cell" else 0.0
        out[c] = out[c].fillna(fill)
    return out
